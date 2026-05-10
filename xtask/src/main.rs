#![recursion_limit = "256"]

use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::io::Write as _;
#[cfg(unix)]
use std::os::unix::fs as unix_fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail, ensure};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::Deserialize;
use serde_json::{Value, json};

const DEFAULT_WASM_MODEL_DIR: &str = "/home/mosure/.cache/huggingface/hub/models--nvidia--AutoGaze/snapshots/5100fae739ec1bf3f875914fa1b703846a18943a";
const BEVY_PERF_CASE_TIMEOUT_SECS: u64 = 600;
const BEVY_PERF_MATRIX_MANIFEST: &str = "matrix.json";

#[derive(Debug, Parser)]
#[command(author, version, about = "burn_autogaze repository task runner")]
struct Cli {
    #[command(subcommand)]
    command: CommandKind,
}

#[derive(Debug, Subcommand)]
enum CommandKind {
    ReleaseReadiness(ReleaseReadinessArgs),
    CheckBevyWasmDemo(CheckBevyWasmDemoArgs),
    CompletionAudit(CompletionAuditArgs),
    BevyPerfMatrix(BevyPerfMatrixArgs),
    ValidateBevyPerfSummary(ValidateBevyPerfSummaryArgs),
    CheckBurnJepaSparseReadoutIntegration(CheckBurnJepaArgs),
    UpstreamFixtureMatrix(UpstreamFixtureMatrixArgs),
}

#[derive(Debug, Clone, Args)]
struct CommonArgs {
    #[arg(long, default_value = "cargo")]
    cargo: String,
    #[arg(long)]
    dry_run: bool,
}

#[derive(Debug, Args)]
struct ReleaseReadinessArgs {
    #[command(flatten)]
    common: CommonArgs,
    #[arg(long)]
    browser: bool,
    #[arg(long)]
    real_model_browser: bool,
    #[arg(long)]
    no_browser_deps: bool,
    #[arg(long)]
    node_bin_dir: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct CheckBevyWasmDemoArgs {
    #[command(flatten)]
    common: CommonArgs,
    #[arg(long)]
    browser: bool,
    #[arg(long)]
    real_model_browser: bool,
    #[arg(long)]
    skip_check: bool,
    #[arg(long)]
    no_browser_deps: bool,
    #[arg(long)]
    node_bin_dir: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct CompletionAuditArgs {
    #[command(flatten)]
    common: CommonArgs,
    #[arg(long)]
    burn_jepa: Option<PathBuf>,
    #[arg(long)]
    hardware_perf: bool,
    #[arg(long)]
    strict: bool,
    #[arg(long, default_value_t = 120)]
    frames: u32,
    #[arg(long, default_value_t = BEVY_PERF_CASE_TIMEOUT_SECS)]
    case_timeout_seconds: u64,
    #[arg(long, value_enum, default_value_t = BevyPerfBuildProfile::Release)]
    perf_profile: BevyPerfBuildProfile,
    #[arg(long, default_value = "target/autogaze-bevy-perf-audit")]
    out: PathBuf,
}

#[derive(Debug, Args)]
struct BevyPerfMatrixArgs {
    #[command(flatten)]
    common: CommonArgs,
    #[arg(long, default_value_t = 120)]
    frames: u32,
    #[arg(
        long,
        default_value = "tests/fixtures/autogaze_birds_python_generate/raw_rgba_frame_00.png"
    )]
    image: PathBuf,
    #[arg(long, default_value = "target/autogaze-bevy-perf")]
    out: PathBuf,
    #[arg(long)]
    camera: bool,
    #[arg(long, default_value_t = BEVY_PERF_CASE_TIMEOUT_SECS)]
    case_timeout_seconds: u64,
    #[arg(long, value_enum, default_value_t = BevyPerfBuildProfile::Release)]
    profile: BevyPerfBuildProfile,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum BevyPerfBuildProfile {
    Release,
    Dev,
}

impl BevyPerfBuildProfile {
    const fn cargo_run_args(self) -> &'static [&'static str] {
        match self {
            Self::Release => &["--release"],
            Self::Dev => &[],
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Release => "release",
            Self::Dev => "dev",
        }
    }
}

#[derive(Debug, Args)]
struct ValidateBevyPerfSummaryArgs {
    path: Option<PathBuf>,
    #[arg(long)]
    require_hardware_adapter: bool,
    #[arg(long)]
    print_summary: bool,
    #[arg(long)]
    self_test: bool,
}

#[derive(Debug, Args)]
struct CheckBurnJepaArgs {
    path: PathBuf,
}

#[derive(Debug, Args)]
struct UpstreamFixtureMatrixArgs {
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long)]
    model_dir: Option<PathBuf>,
    #[arg(long = "case")]
    case_names: Vec<String>,
    #[arg(long)]
    skip_existing: bool,
    #[arg(long)]
    allow_outside_fixtures: bool,
    #[arg(long)]
    run_parity_test: bool,
    #[arg(long, default_value = "cargo")]
    cargo: String,
    #[arg(long)]
    dry_run: bool,
}

struct Runner {
    root: PathBuf,
    cargo: String,
    dry_run: bool,
    path_prefixes: Vec<PathBuf>,
    rustc: Option<PathBuf>,
}

impl Runner {
    fn new(root: PathBuf, common: &CommonArgs, node_bin_dir: Option<PathBuf>) -> Self {
        let mut path_prefixes = Vec::new();
        let nightly_bin =
            PathBuf::from("/home/mosure/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/bin");
        let rustc = if nightly_bin.is_dir() {
            path_prefixes.push(nightly_bin.clone());
            Some(nightly_bin.join("rustc"))
        } else {
            None
        };
        if let Some(path) = node_bin_dir {
            path_prefixes.push(path);
        }
        Self {
            root,
            cargo: common.cargo.clone(),
            dry_run: common.dry_run,
            path_prefixes,
            rustc,
        }
    }

    fn command<I, S>(&self, program: &str, args: I) -> PreparedCommand
    where
        I: IntoIterator<Item = S>,
        S: Into<OsString>,
    {
        PreparedCommand {
            program: OsString::from(program),
            args: args.into_iter().map(Into::into).collect(),
            cwd: self.root.clone(),
            envs: Vec::new(),
        }
    }

    fn command_in<I, S>(&self, cwd: impl Into<PathBuf>, program: &str, args: I) -> PreparedCommand
    where
        I: IntoIterator<Item = S>,
        S: Into<OsString>,
    {
        PreparedCommand {
            program: OsString::from(program),
            args: args.into_iter().map(Into::into).collect(),
            cwd: cwd.into(),
            envs: Vec::new(),
        }
    }

    fn run(&self, command: PreparedCommand) -> Result<()> {
        println!("\n+ {}", command.display());
        if self.dry_run {
            return Ok(());
        }
        let status = self
            .std_command(&command)
            .status()
            .with_context(|| format!("run {}", command.display()))?;
        ensure!(
            status.success(),
            "command failed with {status}: {}",
            command.display()
        );
        Ok(())
    }

    fn output(&self, command: &PreparedCommand) -> Result<std::process::Output> {
        if self.dry_run {
            bail!(
                "cannot capture output in dry-run mode: {}",
                command.display()
            );
        }
        self.std_command(command)
            .output()
            .with_context(|| format!("run {}", command.display()))
    }

    fn output_with_timeout(
        &self,
        command: &PreparedCommand,
        timeout: Duration,
    ) -> Result<TimedOutput> {
        if self.dry_run {
            bail!(
                "cannot capture output in dry-run mode: {}",
                command.display()
            );
        }
        let mut child = self
            .std_command(command)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("run {}", command.display()))?;
        let started = Instant::now();
        loop {
            if child.try_wait()?.is_some() {
                let output = child.wait_with_output()?;
                return Ok(TimedOutput {
                    output,
                    timed_out: false,
                });
            }
            if started.elapsed() >= timeout {
                let _ = child.kill();
                let output = child.wait_with_output()?;
                return Ok(TimedOutput {
                    output,
                    timed_out: true,
                });
            }
            thread::sleep(Duration::from_millis(200));
        }
    }

    fn std_command(&self, command: &PreparedCommand) -> Command {
        let mut std_command = Command::new(&command.program);
        std_command.args(&command.args).current_dir(&command.cwd);
        if let Some(path) = self.composed_path() {
            std_command.env("PATH", path);
        }
        if std::env::var_os("RUSTC").is_none()
            && let Some(rustc) = &self.rustc
        {
            std_command.env("RUSTC", rustc);
        }
        for (key, value) in &command.envs {
            std_command.env(key, value);
        }
        std_command
    }

    fn composed_path(&self) -> Option<OsString> {
        if self.path_prefixes.is_empty() {
            return None;
        }
        let current = std::env::var_os("PATH").unwrap_or_default();
        let mut entries = self.path_prefixes.clone();
        entries.extend(std::env::split_paths(&current));
        std::env::join_paths(entries).ok()
    }
}

struct TimedOutput {
    output: Output,
    timed_out: bool,
}

struct PreparedCommand {
    program: OsString,
    args: Vec<OsString>,
    cwd: PathBuf,
    envs: Vec<(OsString, OsString)>,
}

impl PreparedCommand {
    fn env(mut self, key: impl Into<OsString>, value: impl Into<OsString>) -> Self {
        self.envs.push((key.into(), value.into()));
        self
    }

    fn display(&self) -> String {
        let mut parts = Vec::new();
        if !self.cwd.as_os_str().is_empty() {
            parts.push(format!("(cd {} &&", shell_escape(self.cwd.as_os_str())));
        }
        for (key, value) in &self.envs {
            parts.push(format!("{}={}", key.to_string_lossy(), shell_escape(value)));
        }
        parts.push(shell_escape(&self.program));
        parts.extend(self.args.iter().map(|arg| shell_escape(arg.as_os_str())));
        if !self.cwd.as_os_str().is_empty() {
            parts.push(")".to_owned());
        }
        parts.join(" ")
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let root = workspace_root()?;
    match cli.command {
        CommandKind::ReleaseReadiness(args) => release_readiness(root, args),
        CommandKind::CheckBevyWasmDemo(args) => check_bevy_wasm_demo(root, args),
        CommandKind::CompletionAudit(args) => completion_audit(root, args),
        CommandKind::BevyPerfMatrix(args) => bevy_perf_matrix(root, args),
        CommandKind::ValidateBevyPerfSummary(args) => validate_bevy_perf_summary_cmd(root, args),
        CommandKind::CheckBurnJepaSparseReadoutIntegration(args) => {
            check_burn_jepa_sparse_readout_integration(&args.path)
        }
        CommandKind::UpstreamFixtureMatrix(args) => upstream_fixture_matrix(root, args),
    }
}

fn release_readiness(root: PathBuf, args: ReleaseReadinessArgs) -> Result<()> {
    let runner = Runner::new(root.clone(), &args.common, args.node_bin_dir.clone());
    let cargo = runner.cargo.clone();
    for command in [
        vec!["test", "-p", "burn_autogaze", "--features", "ndarray"],
        vec!["test", "-p", "bevy_burn_autogaze"],
        vec![
            "clippy",
            "-p",
            "burn_autogaze",
            "--features",
            "ndarray",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ],
        vec![
            "clippy",
            "-p",
            "bevy_burn_autogaze",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ],
        vec![
            "check",
            "-p",
            "burn_autogaze",
            "--target",
            "wasm32-unknown-unknown",
            "--no-default-features",
            "--features",
            "wasm",
        ],
        vec![
            "clippy",
            "-p",
            "burn_autogaze",
            "--target",
            "wasm32-unknown-unknown",
            "--no-default-features",
            "--features",
            "wasm",
            "--",
            "-D",
            "warnings",
        ],
        vec![
            "check",
            "-p",
            "bevy_burn_autogaze",
            "--target",
            "wasm32-unknown-unknown",
        ],
        vec![
            "clippy",
            "-p",
            "bevy_burn_autogaze",
            "--target",
            "wasm32-unknown-unknown",
            "--",
            "-D",
            "warnings",
        ],
        vec![
            "check",
            "--example",
            "sparse_video_readout_adapter",
            "--features",
            "ndarray",
        ],
    ] {
        runner.run(runner.command(&cargo, command))?;
    }

    runner.run(runner.command(&cargo, ["check", "-p", "xtask"]))?;
    runner.run(runner.command(&cargo, ["test", "-p", "xtask"]))?;
    runner.run(runner.command(&cargo, ["clippy", "-p", "xtask", "--", "-D", "warnings"]))?;
    completion_audit(
        root.clone(),
        CompletionAuditArgs {
            common: CommonArgs {
                cargo: cargo.clone(),
                dry_run: args.common.dry_run,
            },
            burn_jepa: None,
            hardware_perf: false,
            strict: false,
            frames: 120,
            case_timeout_seconds: BEVY_PERF_CASE_TIMEOUT_SECS,
            perf_profile: BevyPerfBuildProfile::Release,
            out: PathBuf::from("target/autogaze-bevy-perf-audit"),
        },
    )?;
    upstream_fixture_matrix(
        root.clone(),
        UpstreamFixtureMatrixArgs {
            manifest: PathBuf::from("docs/upstream_fixture_matrix.example.json"),
            model_dir: None,
            case_names: Vec::new(),
            skip_existing: false,
            allow_outside_fixtures: false,
            run_parity_test: false,
            cargo: cargo.clone(),
            dry_run: true,
        },
    )?;
    validate_bevy_perf_summary_self_test()?;
    bevy_perf_matrix(
        root.clone(),
        BevyPerfMatrixArgs {
            common: CommonArgs {
                cargo: cargo.clone(),
                dry_run: true,
            },
            frames: 2,
            image: PathBuf::from(
                "tests/fixtures/autogaze_birds_python_generate/raw_rgba_frame_00.png",
            ),
            out: PathBuf::from("target/autogaze-bevy-perf"),
            camera: true,
            case_timeout_seconds: BEVY_PERF_CASE_TIMEOUT_SECS,
            profile: BevyPerfBuildProfile::Release,
        },
    )?;

    for command in [
        vec![
            "bench",
            "--bench",
            "backend_pipeline",
            "--features",
            "ndarray",
            "--no-run",
        ],
        vec![
            "bench",
            "-p",
            "bevy_burn_autogaze",
            "--bench",
            "viewer_pipeline",
            "--no-run",
        ],
        vec!["package", "-p", "burn_autogaze", "--allow-dirty"],
    ] {
        runner.run(runner.command(&cargo, command))?;
    }

    let package_dir = package_dir(&root)?;
    if !args.common.dry_run {
        ensure!(
            package_dir.is_dir(),
            "expected generated package checkout missing: {}",
            package_dir.display()
        );
    }
    for command in [
        vec![
            "test",
            "--features",
            "ndarray",
            "--test",
            "source_hygiene",
            "--",
            "--nocapture",
        ],
        vec![
            "test",
            "--features",
            "ndarray",
            "--test",
            "native_autogaze_generate_parity",
            "upstream_generated_masks_decode_without_model_snapshot",
            "--",
            "--nocapture",
        ],
        vec![
            "test",
            "--features",
            "ndarray",
            "metrics",
            "--",
            "--nocapture",
        ],
        vec![
            "test",
            "--features",
            "ndarray",
            "readout",
            "--",
            "--nocapture",
        ],
        vec![
            "check",
            "--example",
            "sparse_video_readout_adapter",
            "--features",
            "ndarray",
        ],
    ] {
        runner.run(runner.command_in(&package_dir, &cargo, command))?;
    }

    if args.browser || args.real_model_browser {
        check_bevy_wasm_demo(
            root.clone(),
            CheckBevyWasmDemoArgs {
                common: CommonArgs {
                    cargo: cargo.clone(),
                    dry_run: args.common.dry_run,
                },
                browser: true,
                real_model_browser: args.real_model_browser,
                skip_check: true,
                no_browser_deps: args.no_browser_deps,
                node_bin_dir: args.node_bin_dir,
            },
        )?;
    }

    runner.run(runner.command("git", ["diff", "--check"]))?;
    Ok(())
}

fn completion_audit(root: PathBuf, args: CompletionAuditArgs) -> Result<()> {
    validate_completion_audit_args(&args)?;
    let runner = Runner::new(root.clone(), &args.common, None);
    let cargo = runner.cargo.clone();
    for command in [
        vec![
            "test",
            "-p",
            "burn_autogaze",
            "--features",
            "ndarray",
            "--test",
            "source_hygiene",
            "--",
            "--nocapture",
        ],
        vec![
            "test",
            "-p",
            "burn_autogaze",
            "--features",
            "ndarray",
            "readout",
            "--",
            "--nocapture",
        ],
        vec![
            "test",
            "-p",
            "burn_autogaze",
            "--features",
            "ndarray",
            "--test",
            "native_autogaze_generate_parity",
            "upstream_generated_masks_decode_without_model_snapshot",
            "--",
            "--nocapture",
        ],
    ] {
        runner.run(runner.command(&cargo, command))?;
    }
    validate_bevy_perf_summary_self_test()?;

    let mut evidence_errors = Vec::new();
    if let Some(path) = &args.burn_jepa {
        if let Err(error) = check_burn_jepa_sparse_readout_integration(path) {
            evidence_errors.push(format!("burn_jepa migration audit failed:\n{error:#}"));
        }
    } else {
        println!(
            "\nskipping burn_jepa migration audit; pass --burn-jepa PATH to enforce that the\nsibling checkout no longer duplicates AutoGaze generated-token decoding."
        );
    }

    if args.hardware_perf {
        let perf_result = bevy_perf_matrix(
            root,
            BevyPerfMatrixArgs {
                common: args.common.clone(),
                frames: args.frames,
                image: PathBuf::from(
                    "tests/fixtures/autogaze_birds_python_generate/raw_rgba_frame_00.png",
                ),
                out: args.out.clone(),
                camera: true,
                case_timeout_seconds: args.case_timeout_seconds,
                profile: args.perf_profile,
            },
        );
        if let Err(error) = perf_result {
            evidence_errors.push(format!(
                "native hardware Bevy perf audit failed:\n{error:#}"
            ));
        }
    } else {
        println!(
            "\nskipping native hardware Bevy perf audit; pass --hardware-perf on a host with a\nreal GPU render adapter and camera to enforce end-to-end throughput evidence."
        );
    }
    finish_completion_audit_evidence(evidence_errors)
}

fn validate_completion_audit_args(args: &CompletionAuditArgs) -> Result<()> {
    ensure!(args.frames > 0, "--frames must be greater than zero");
    ensure!(
        args.case_timeout_seconds > 0,
        "--case-timeout-seconds must be greater than zero"
    );
    if args.strict {
        ensure!(
            args.burn_jepa.is_some(),
            "strict completion audit requires --burn-jepa PATH"
        );
        ensure!(
            args.hardware_perf,
            "strict completion audit requires --hardware-perf on a real GPU/camera host"
        );
    }
    Ok(())
}

fn finish_completion_audit_evidence(errors: Vec<String>) -> Result<()> {
    if errors.is_empty() {
        return Ok(());
    }

    let report = errors
        .iter()
        .enumerate()
        .map(|(index, error)| format!("{}. {error}", index + 1))
        .collect::<Vec<_>>()
        .join("\n\n");
    bail!("completion audit evidence failed:\n{report}")
}

fn check_bevy_wasm_demo(root: PathBuf, args: CheckBevyWasmDemoArgs) -> Result<()> {
    let runner = Runner::new(root.clone(), &args.common, args.node_bin_dir);
    let cargo = runner.cargo.clone();
    let bevy_dir = root.join("crates/bevy_burn_autogaze");
    if !args.skip_check {
        runner.run(runner.command(
            &cargo,
            [
                "check",
                "-p",
                "bevy_burn_autogaze",
                "--target",
                "wasm32-unknown-unknown",
            ],
        ))?;
    }

    install_wasm_bindgen_cli(&runner)?;
    require_node_toolchain(&runner)?;
    runner.run(runner.command_in(&bevy_dir, "npm", ["ci"]))?;
    runner.run(runner.command_in(&bevy_dir, "npm", ["run", "build:wasm"]))?;

    if args.browser {
        let install_args = if args.no_browser_deps {
            vec!["playwright", "install", "chromium"]
        } else {
            vec!["playwright", "install", "--with-deps", "chromium"]
        };
        runner.run(runner.command_in(&bevy_dir, "npx", install_args))?;
        runner.run(runner.command_in(&bevy_dir, "npm", ["run", "test:browser"]))?;
    }

    if args.real_model_browser {
        let staged = stage_wasm_model_assets(&runner, &root)?;
        let result = runner.run(
            runner
                .command_in(&bevy_dir, "npm", ["run", "test:browser"])
                .env("AUTOGAZE_WASM_MODEL_E2E", "1"),
        );
        cleanup_paths(staged)?;
        result?;
    }
    Ok(())
}

fn install_wasm_bindgen_cli(runner: &Runner) -> Result<()> {
    let version = wasm_bindgen_version(&runner.root)?;
    if !runner.dry_run {
        let command = runner.command("wasm-bindgen", ["--version"]);
        if let Ok(output) = runner.output(&command) {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if output.status.success() && stdout.split_whitespace().nth(1) == Some(version.as_str())
            {
                println!("wasm-bindgen-cli {version} already installed");
                return Ok(());
            }
        }
    }
    runner.run(runner.command(
        &runner.cargo,
        [
            "install",
            "wasm-bindgen-cli",
            "--version",
            &version,
            "--locked",
        ],
    ))
}

fn require_node_toolchain(runner: &Runner) -> Result<()> {
    if runner.dry_run {
        runner.run(runner.command("node", ["--version"]))?;
        runner.run(runner.command("npm", ["--version"]))?;
        runner.run(runner.command("npx", ["--version"]))?;
        return Ok(());
    }
    let mut output_text = String::new();
    for tool in ["node", "npm", "npx"] {
        let output = runner.output(&runner.command(tool, ["--version"]));
        match output {
            Ok(output) if output.status.success() => {
                output_text.push_str(tool);
                output_text.push_str(": ");
                output_text.push_str(String::from_utf8_lossy(&output.stdout).trim());
                output_text.push('\n');
            }
            Ok(output) => {
                output_text.push_str(&String::from_utf8_lossy(&output.stderr));
                output_text.push_str(&String::from_utf8_lossy(&output.stdout));
                node_toolchain_error(&output_text)?;
            }
            Err(error) => bail!("missing required browser-test tool `{tool}`: {error:#}"),
        }
    }
    println!("node toolchain:\n{output_text}");
    Ok(())
}

fn node_toolchain_error(output: &str) -> Result<()> {
    if output.to_ascii_lowercase().contains("snap-confine") {
        bail!(
            "node/npm/npx preflight failed:\n{output}\nThe active Node.js toolchain appears to be Snap-provided and cannot run in this environment.\nUse a non-Snap Node.js install, set AUTOGAZE_NODE_BIN_DIR, or pass --node-bin-dir PATH."
        );
    }
    bail!("node/npm/npx preflight failed:\n{output}")
}

fn stage_wasm_model_assets(runner: &Runner, root: &Path) -> Result<Vec<PathBuf>> {
    let www_dir = root.join("crates/bevy_burn_autogaze/www");
    let model_dir = std::env::var_os("AUTOGAZE_WASM_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_WASM_MODEL_DIR));
    let config_src = std::env::var_os("AUTOGAZE_WASM_CONFIG")
        .map(PathBuf::from)
        .unwrap_or_else(|| model_dir.join("config.json"));
    let weights_src = std::env::var_os("AUTOGAZE_WASM_WEIGHTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| model_dir.join("model.safetensors"));
    let config_dst = www_dir.join("config.json");
    let weights_dst = www_dir.join("model.safetensors");

    if config_dst.exists() && weights_dst.exists() {
        return Ok(Vec::new());
    }
    if !config_src.is_file() || !weights_src.is_file() {
        eprintln!(
            "real-model browser assets not staged; missing {} or {}",
            config_src.display(),
            weights_src.display()
        );
        eprintln!(
            "Set AUTOGAZE_WASM_MODEL_DIR, AUTOGAZE_WASM_CONFIG, or AUTOGAZE_WASM_WEIGHTS to enable the real-model browser smoke."
        );
        return Ok(Vec::new());
    }

    let mut staged = Vec::new();
    for (src, dst) in [(config_src, config_dst), (weights_src, weights_dst)] {
        if !dst.exists() {
            println!("\n+ ln -s {} {}", src.display(), dst.display());
            if !runner.dry_run {
                #[cfg(unix)]
                unix_fs::symlink(&src, &dst)
                    .with_context(|| format!("symlink {} -> {}", dst.display(), src.display()))?;
                #[cfg(not(unix))]
                fs::copy(&src, &dst)
                    .with_context(|| format!("copy {} -> {}", src.display(), dst.display()))?;
            }
            staged.push(dst);
        }
    }
    Ok(staged)
}

fn cleanup_paths(paths: Vec<PathBuf>) -> Result<()> {
    for path in paths {
        if path.exists() || path.is_symlink() {
            fs::remove_file(&path).with_context(|| format!("remove {}", path.display()))?;
        }
    }
    Ok(())
}

fn bevy_perf_matrix(root: PathBuf, args: BevyPerfMatrixArgs) -> Result<()> {
    ensure!(args.frames > 0, "--frames must be greater than zero");
    ensure!(
        args.case_timeout_seconds > 0,
        "--case-timeout-seconds must be greater than zero"
    );
    let runner = Runner::new(root, &args.common, None);
    let common_static = [
        "--image-path".to_owned(),
        args.image.display().to_string(),
        "--show-psnr=false".to_owned(),
    ];
    let mut cases = vec![
        (
            "realtime-static-cpu",
            vec![
                "--mode",
                "realtime",
                "--display-transfer",
                "cpu",
                "--visualization-mode",
                "full-blend",
            ],
            true,
        ),
        (
            "realtime-static-gpu",
            vec![
                "--mode",
                "realtime",
                "--display-transfer",
                "gpu",
                "--visualization-mode",
                "full-blend",
            ],
            true,
        ),
        (
            "realtime-static-interframe",
            vec![
                "--mode",
                "realtime",
                "--display-transfer",
                "cpu",
                "--visualization-mode",
                "interframe",
            ],
            true,
        ),
        (
            "tiled-static-interframe",
            vec![
                "--mode",
                "tiled",
                "--display-transfer",
                "cpu",
                "--visualization-mode",
                "interframe",
            ],
            true,
        ),
    ];
    if args.camera {
        cases.extend([
            (
                "realtime-camera",
                vec![
                    "--mode",
                    "realtime",
                    "--display-transfer",
                    "cpu",
                    "--visualization-mode",
                    "full-blend",
                ],
                false,
            ),
            (
                "tiled-camera-interframe",
                vec![
                    "--mode",
                    "tiled",
                    "--display-transfer",
                    "cpu",
                    "--visualization-mode",
                    "interframe",
                ],
                false,
            ),
        ]);
    }

    if !runner.dry_run {
        write_perf_matrix_manifest(&args.out, &args, &cases)?;
    }

    for (name, case_args, include_static) in cases {
        let mut app_args = Vec::<OsString>::new();
        if include_static {
            app_args.extend(common_static.iter().cloned().map(OsString::from));
        }
        app_args.extend(case_args.into_iter().map(OsString::from));
        app_args.extend([
            OsString::from("--perf-summary-frames"),
            OsString::from(args.frames.to_string()),
            OsString::from("--perf-summary-path"),
            OsString::from(args.out.join(format!("{name}.json"))),
            OsString::from("--log-pipeline-timing"),
            OsString::from("--require-hardware-adapter=true"),
        ]);
        run_bevy_perf_case(
            &runner,
            &args.out,
            name,
            app_args,
            Duration::from_secs(args.case_timeout_seconds),
            args.profile,
        )?;
    }
    if !runner.dry_run {
        write_perf_aggregate(&args.out)?;
        let summary = load_json(&args.out.join("summary.json"))?;
        validate_summary_or_aggregate(&summary, true)?;
        print_perf_summary(&summary)?;
        println!("\nwrote logs and JSON summaries to {}", args.out.display());
    }
    Ok(())
}

fn write_perf_matrix_manifest(
    out_dir: &Path,
    args: &BevyPerfMatrixArgs,
    cases: &[(&str, Vec<&str>, bool)],
) -> Result<()> {
    fs::create_dir_all(out_dir)?;
    let cases = cases
        .iter()
        .map(|(name, app_args, include_static)| {
            json!({
                "case": name,
                "include_static_source": include_static,
                "app_args": app_args,
                "summary_path": out_dir.join(format!("{name}.json")).display().to_string(),
                "log_path": out_dir.join(format!("{name}.log")).display().to_string(),
            })
        })
        .collect::<Vec<_>>();
    let manifest = json!({
        "frames": args.frames,
        "image": args.image.display().to_string(),
        "camera": args.camera,
        "xtask_build_profile": args.profile.as_str(),
        "xtask_case_timeout_seconds": args.case_timeout_seconds,
        "xtask_cache_dir": out_dir.join("cache").display().to_string(),
        "require_hardware_adapter": true,
        "cases": cases,
    });
    write_json(&out_dir.join(BEVY_PERF_MATRIX_MANIFEST), &manifest)
}

fn run_bevy_perf_case(
    runner: &Runner,
    out_dir: &Path,
    name: &str,
    app_args: Vec<OsString>,
    timeout: Duration,
    profile: BevyPerfBuildProfile,
) -> Result<()> {
    let log_path = out_dir.join(format!("{name}.log"));
    let json_path = out_dir.join(format!("{name}.json"));
    let mut args = vec![
        OsString::from("run"),
        OsString::from("-p"),
        OsString::from("bevy_burn_autogaze"),
    ];
    args.extend(profile.cargo_run_args().iter().copied().map(OsString::from));
    args.push(OsString::from("--"));
    args.extend(app_args);
    let command = runner
        .command(&runner.cargo, args)
        .env("XDG_CACHE_HOME", out_dir.join("cache"));
    println!("\n[{name}]");
    println!("  {}", command.display());
    if runner.dry_run {
        return Ok(());
    }
    fs::create_dir_all(out_dir)?;
    let timed_output = runner
        .output_with_timeout(&command, timeout)
        .with_context(|| format!("run case {name}"))?;
    let output = timed_output.output;
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    print!("{combined}");
    fs::write(&log_path, &combined)?;
    ensure!(
        !timed_output.timed_out,
        "case {name} timed out after {}s; see {}",
        timeout.as_secs(),
        log_path.display()
    );
    ensure!(
        output.status.success(),
        "case {name} failed with status {}; see {}",
        output.status,
        log_path.display()
    );
    normalize_perf_json(&log_path, &json_path)?;
    let mut data = load_json(&json_path)?;
    augment_perf_case_metadata(&mut data, name, profile, timeout, &out_dir.join("cache"))?;
    write_json(&json_path, &data)?;
    validate_summary_or_aggregate(&data, true)?;
    print_perf_summary(&data)?;
    Ok(())
}

fn augment_perf_case_metadata(
    data: &mut Value,
    name: &str,
    profile: BevyPerfBuildProfile,
    timeout: Duration,
    cache_dir: &Path,
) -> Result<()> {
    let object = data
        .as_object_mut()
        .ok_or_else(|| anyhow!("perf summary must be a JSON object"))?;
    object.insert("case".to_owned(), Value::String(name.to_owned()));
    object.insert(
        "xtask_build_profile".to_owned(),
        Value::String(profile.as_str().to_owned()),
    );
    object.insert(
        "xtask_case_timeout_seconds".to_owned(),
        Value::from(timeout.as_secs()),
    );
    object.insert(
        "xtask_cache_dir".to_owned(),
        Value::String(cache_dir.display().to_string()),
    );
    Ok(())
}

fn normalize_perf_json(log_path: &Path, json_path: &Path) -> Result<()> {
    let data = if json_path.is_file() {
        load_json(json_path)?
    } else {
        let log = fs::read_to_string(log_path)?;
        let prefix = "AutoGaze perf summary:";
        let summary = log
            .lines()
            .find_map(|line| line.split_once(prefix).map(|(_, value)| value.trim()))
            .ok_or_else(|| anyhow!("no AutoGaze perf summary found in {}", log_path.display()))?;
        serde_json::from_str(summary).context("parse perf summary from log")?
    };
    write_json(json_path, &data)
}

fn write_json(path: &Path, data: &Value) -> Result<()> {
    let mut file = File::create(path)?;
    serde_json::to_writer_pretty(&mut file, &data)?;
    writeln!(file)?;
    Ok(())
}

fn write_perf_aggregate(out_dir: &Path) -> Result<()> {
    let mut rows = Vec::new();
    for entry in fs::read_dir(out_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !is_perf_case_json_path(&path) {
            continue;
        }
        let mut row = load_json(&path)?;
        if let Some(object) = row.as_object_mut() {
            object.insert(
                "case".to_owned(),
                Value::String(
                    path.file_stem()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into(),
                ),
            );
        }
        rows.push(row);
    }
    rows.sort_by_key(|row| {
        row.get("case")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned()
    });
    ensure!(
        !rows.is_empty(),
        "no per-case JSON summaries found in {}",
        out_dir.display()
    );
    let fps_values = rows
        .iter()
        .filter_map(|row| row.get("avg_output_fps").and_then(Value::as_f64))
        .collect::<Vec<_>>();
    let summary = json!({
        "case_count": rows.len(),
        "min_output_fps": fps_values.iter().copied().reduce(f64::min).unwrap_or(0.0),
        "max_output_fps": fps_values.iter().copied().reduce(f64::max).unwrap_or(0.0),
        "cases": rows,
    });
    let mut file = File::create(out_dir.join("summary.json"))?;
    serde_json::to_writer_pretty(&mut file, &summary)?;
    writeln!(file)?;
    Ok(())
}

fn is_perf_case_json_path(path: &Path) -> bool {
    path.extension() == Some(OsStr::new("json"))
        && !matches!(
            path.file_name(),
            Some(name) if name == OsStr::new("summary.json")
                || name == OsStr::new(BEVY_PERF_MATRIX_MANIFEST)
        )
}

fn validate_bevy_perf_summary_cmd(root: PathBuf, args: ValidateBevyPerfSummaryArgs) -> Result<()> {
    let _ = root;
    if args.self_test {
        validate_bevy_perf_summary_self_test()?;
    }
    if let Some(path) = args.path {
        let data = load_json(&path)?;
        validate_summary_or_aggregate(&data, args.require_hardware_adapter)?;
        if args.print_summary {
            print_perf_summary(&data)?;
        }
    } else if !args.self_test {
        bail!("provide a summary path or --self-test");
    }
    Ok(())
}

fn validate_summary_or_aggregate(data: &Value, require_hardware_adapter: bool) -> Result<()> {
    let object = data
        .as_object()
        .ok_or_else(|| anyhow!("summary must be a JSON object"))?;
    if object.contains_key("cases") {
        validate_aggregate_summary(data, require_hardware_adapter)
    } else {
        validate_summary(data, require_hardware_adapter)
    }
}

fn validate_summary(data: &Value, require_hardware_adapter: bool) -> Result<()> {
    for field in [
        "avg_output_fps",
        "avg_model_frame_fps",
        "avg_input_fps",
        "p95_total_ms",
    ] {
        require_number(data, field, 0.0, None, true)?;
    }
    for field in [
        "avg_total_ms",
        "p50_total_ms",
        "avg_model_ms",
        "avg_input_ms",
        "avg_pack_ms",
        "avg_visualize_ms",
        "avg_visualize_cpu_ms",
        "avg_tensor_ms",
        "avg_display_ms",
    ] {
        require_number(data, field, 0.0, None, false)?;
    }
    for field in ["avg_gaze_update_ratio", "latest_gaze_update_ratio"] {
        require_number(data, field, 0.0, Some(1.0), true)?;
    }
    for field in [
        "processed_frames",
        "latest_clip_frames",
        "latest_model_frames",
        "latest_width",
        "latest_height",
    ] {
        require_int(data, field, 1, true)?;
    }
    for field in [
        "processed_model_frames",
        "latest_trace_points",
        "latest_sequence",
    ] {
        require_int(data, field, 0, true)?;
    }
    if let Some(target_frames) = require_int(data, "target_frames", 1, false)? {
        let processed = require_required_int(data, "processed_frames", 1)?;
        ensure!(
            processed <= target_frames,
            "`processed_frames` must not exceed `target_frames`: {processed} > {target_frames}"
        );
    }
    let p50 = require_number(data, "p50_total_ms", 0.0, None, false)?;
    let p95 = require_required_number(data, "p95_total_ms", 0.0, None)?;
    if let Some(p50) = p50 {
        ensure!(
            p95 >= p50,
            "`p95_total_ms` must be >= `p50_total_ms`: {p95} < {p50}"
        );
    }
    require_enum(data, "mode", &["resize-224", "tiled"], true)?;
    require_enum(
        data,
        "visualization_mode",
        &["full-blend", "interframe"],
        false,
    )?;
    require_enum(data, "display_transfer", &["cpu", "gpu"], false)?;
    require_enum(data, "xtask_build_profile", &["release", "dev"], false)?;
    require_int(data, "xtask_case_timeout_seconds", 1, false)?;
    require_string(data, "xtask_cache_dir", false)?;
    require_nullable_enum(
        data,
        "latest_tensor_interframe_path",
        &["dense-tensor", "sparse-rects"],
    )?;
    for field in ["streaming_cache", "streaming_cache_effective"] {
        require_bool(data, field, false)?;
    }
    let latest_psnr = require_nullable_number(data, "latest_psnr_db", 0.0)?;
    let latest_inf = require_required_bool(data, "latest_psnr_db_infinite")?;
    let ema_psnr = require_nullable_number(data, "ema_psnr_db", 0.0)?;
    let ema_inf = require_required_bool(data, "ema_psnr_db_infinite")?;
    let show_psnr = require_bool(data, "show_psnr", false)?.unwrap_or(false);
    ensure!(
        !(latest_inf && latest_psnr.is_some()),
        "`latest_psnr_db` must be null when `latest_psnr_db_infinite` is true"
    );
    ensure!(
        !(ema_inf && ema_psnr.is_some()),
        "`ema_psnr_db` must be null when `ema_psnr_db_infinite` is true"
    );
    if show_psnr {
        ensure!(
            latest_psnr.is_some() || latest_inf,
            "PSNR is enabled but latest PSNR is neither finite nor infinite"
        );
        ensure!(
            ema_psnr.is_some() || ema_inf,
            "PSNR is enabled but EMA PSNR is neither finite nor infinite"
        );
    }
    for field in [
        "configured_max_in_flight",
        "effective_max_in_flight",
        "frames_per_clip",
        "top_k",
        "tile_batch_size",
        "inference_width",
        "inference_height",
    ] {
        require_int(data, field, 1, false)?;
    }
    require_int(data, "max_gaze_tokens_each_frame", 0, false)?;
    require_int(data, "tensor_sparse_update_max_rects", 0, false)?;
    require_number(
        data,
        "tensor_sparse_update_max_ratio",
        0.0,
        Some(1.0),
        false,
    )?;
    if require_hardware_adapter {
        let adapter_type = data
            .get("render_adapter_device_type")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                anyhow!("hardware-adapter validation requires `render_adapter_device_type`")
            })?;
        ensure!(
            !adapter_type.eq_ignore_ascii_case("cpu"),
            "render adapter was CPU: {}",
            data.get("render_adapter_name")
                .and_then(Value::as_str)
                .unwrap_or("unknown adapter")
        );
    }
    Ok(())
}

fn validate_aggregate_summary(data: &Value, require_hardware_adapter: bool) -> Result<()> {
    let case_count = usize::try_from(require_required_int(data, "case_count", 1)?)
        .context("case_count exceeds usize")?;
    let min_output_fps = require_required_number(data, "min_output_fps", 0.0, None)?;
    let max_output_fps = require_required_number(data, "max_output_fps", 0.0, None)?;
    ensure!(
        min_output_fps <= max_output_fps,
        "`min_output_fps` must be <= `max_output_fps`"
    );
    let cases = data
        .get("cases")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("`cases` must be an array"))?;
    ensure!(
        cases.len() == case_count,
        "`case_count` does not match cases length"
    );
    let mut fps_values = Vec::new();
    for case in cases {
        validate_summary(case, require_hardware_adapter)?;
        let name = case
            .get("case")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("aggregate case is missing `case` name"))?;
        ensure!(!name.is_empty(), "aggregate case name must not be empty");
        fps_values.push(require_required_number(case, "avg_output_fps", 0.0, None)?);
    }
    let observed_min = fps_values.iter().copied().reduce(f64::min).unwrap_or(0.0);
    let observed_max = fps_values.iter().copied().reduce(f64::max).unwrap_or(0.0);
    ensure!(
        (observed_min - min_output_fps).abs() <= f64::EPSILON,
        "`min_output_fps` does not match cases"
    );
    ensure!(
        (observed_max - max_output_fps).abs() <= f64::EPSILON,
        "`max_output_fps` does not match cases"
    );
    Ok(())
}

fn validate_bevy_perf_summary_self_test() -> Result<()> {
    let sample = json!({
        "avg_output_fps": 54.0,
        "avg_model_frame_fps": 54.0,
        "avg_input_fps": 60.0,
        "avg_total_ms": 18.0,
        "p50_total_ms": 17.0,
        "p95_total_ms": 19.0,
        "avg_model_ms": 6.0,
        "avg_input_ms": 1.0,
        "avg_pack_ms": 1.0,
        "avg_visualize_ms": 2.0,
        "avg_visualize_cpu_ms": 1.0,
        "avg_tensor_ms": 1.0,
        "avg_display_ms": 1.0,
        "avg_gaze_update_ratio": 0.25,
        "latest_gaze_update_ratio": 0.2,
        "processed_frames": 4,
        "processed_model_frames": 8,
        "latest_clip_frames": 2,
        "latest_model_frames": 2,
        "latest_width": 640,
        "latest_height": 360,
        "latest_trace_points": 12,
        "latest_sequence": 3,
        "target_frames": 4,
        "mode": "resize-224",
        "visualization_mode": "interframe",
        "display_transfer": "gpu",
        "xtask_build_profile": "release",
        "xtask_case_timeout_seconds": 600,
        "xtask_cache_dir": "target/autogaze-bevy-perf/cache",
        "latest_tensor_interframe_path": "sparse-rects",
        "streaming_cache": true,
        "streaming_cache_effective": true,
        "latest_psnr_db": 35.0,
        "latest_psnr_db_infinite": false,
        "ema_psnr_db": 34.0,
        "ema_psnr_db_infinite": false,
        "show_psnr": true,
        "configured_max_in_flight": 1,
        "effective_max_in_flight": 1,
        "frames_per_clip": 2,
        "top_k": 10,
        "tile_batch_size": 64,
        "inference_width": 640,
        "inference_height": 360,
        "max_gaze_tokens_each_frame": 0,
        "tensor_sparse_update_max_rects": 256,
        "tensor_sparse_update_max_ratio": 0.35,
        "render_adapter_device_type": "DiscreteGpu",
        "render_adapter_name": "test gpu",
        "case": "self-test"
    });
    validate_summary(&sample, true)?;
    let aggregate = json!({
        "case_count": 1,
        "min_output_fps": 54.0,
        "max_output_fps": 54.0,
        "cases": [sample],
    });
    validate_aggregate_summary(&aggregate, true)?;
    Ok(())
}

fn require_number(
    data: &Value,
    field: &str,
    minimum: f64,
    maximum: Option<f64>,
    required: bool,
) -> Result<Option<f64>> {
    let Some(value) = data.get(field) else {
        ensure!(!required, "missing required numeric field `{field}`");
        return Ok(None);
    };
    let value = value
        .as_f64()
        .ok_or_else(|| anyhow!("`{field}` must be numeric"))?;
    ensure!(value.is_finite(), "`{field}` must be finite");
    ensure!(
        value >= minimum,
        "`{field}` must be >= {minimum}, got {value}"
    );
    if let Some(maximum) = maximum {
        ensure!(
            value <= maximum,
            "`{field}` must be <= {maximum}, got {value}"
        );
    }
    Ok(Some(value))
}

fn require_required_number(
    data: &Value,
    field: &str,
    minimum: f64,
    maximum: Option<f64>,
) -> Result<f64> {
    require_number(data, field, minimum, maximum, true)?
        .ok_or_else(|| anyhow!("missing required numeric field `{field}`"))
}

fn require_nullable_number(data: &Value, field: &str, minimum: f64) -> Result<Option<f64>> {
    let value = data
        .get(field)
        .ok_or_else(|| anyhow!("missing required nullable numeric field `{field}`"))?;
    if value.is_null() {
        return Ok(None);
    }
    require_number(data, field, minimum, None, true)
}

fn require_int(data: &Value, field: &str, minimum: u64, required: bool) -> Result<Option<u64>> {
    let Some(value) = data.get(field) else {
        ensure!(!required, "missing required integer field `{field}`");
        return Ok(None);
    };
    let value = value
        .as_u64()
        .ok_or_else(|| anyhow!("`{field}` must be a nonnegative integer"))?;
    ensure!(
        value >= minimum,
        "`{field}` must be >= {minimum}, got {value}"
    );
    Ok(Some(value))
}

fn require_required_int(data: &Value, field: &str, minimum: u64) -> Result<u64> {
    require_int(data, field, minimum, true)?
        .ok_or_else(|| anyhow!("missing required integer field `{field}`"))
}

fn require_bool(data: &Value, field: &str, required: bool) -> Result<Option<bool>> {
    let Some(value) = data.get(field) else {
        ensure!(!required, "missing required boolean field `{field}`");
        return Ok(None);
    };
    let value = value
        .as_bool()
        .ok_or_else(|| anyhow!("`{field}` must be a boolean"))?;
    Ok(Some(value))
}

fn require_required_bool(data: &Value, field: &str) -> Result<bool> {
    require_bool(data, field, true)?
        .ok_or_else(|| anyhow!("missing required boolean field `{field}`"))
}

fn require_enum(data: &Value, field: &str, allowed: &[&str], required: bool) -> Result<()> {
    let Some(value) = data.get(field) else {
        ensure!(!required, "missing required enum field `{field}`");
        return Ok(());
    };
    let value = value
        .as_str()
        .ok_or_else(|| anyhow!("`{field}` must be a string"))?;
    ensure!(
        allowed.contains(&value),
        "`{field}` must be one of {:?}, got {value:?}",
        allowed
    );
    Ok(())
}

fn require_string(data: &Value, field: &str, required: bool) -> Result<Option<String>> {
    let Some(value) = data.get(field) else {
        ensure!(!required, "missing required string field `{field}`");
        return Ok(None);
    };
    let value = value
        .as_str()
        .ok_or_else(|| anyhow!("`{field}` must be a string"))?;
    ensure!(!value.is_empty(), "`{field}` must not be empty");
    Ok(Some(value.to_owned()))
}

fn require_nullable_enum(data: &Value, field: &str, allowed: &[&str]) -> Result<()> {
    let Some(value) = data.get(field) else {
        return Ok(());
    };
    if value.is_null() {
        return Ok(());
    }
    let value = value
        .as_str()
        .ok_or_else(|| anyhow!("`{field}` must be a string or null"))?;
    ensure!(
        allowed.contains(&value),
        "`{field}` must be one of {:?} or null, got {value:?}",
        allowed
    );
    Ok(())
}

fn print_perf_summary(data: &Value) -> Result<()> {
    if let Some(cases) = data.get("cases").and_then(Value::as_array) {
        println!("aggregate summary:");
        for row in cases {
            print_perf_row(row)?;
        }
    } else {
        print_perf_row(data)?;
    }
    Ok(())
}

fn print_perf_row(row: &Value) -> Result<()> {
    let case = row.get("case").and_then(Value::as_str).unwrap_or("summary");
    let fps = row
        .get("avg_output_fps")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let total = row
        .get("avg_total_ms")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let model = row
        .get("avg_model_ms")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let gaze = row
        .get("avg_gaze_update_ratio")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let adapter = row
        .get("render_adapter_name")
        .and_then(Value::as_str)
        .unwrap_or("unknown adapter");
    let psnr = if row
        .get("latest_psnr_db_infinite")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        "inf".to_owned()
    } else if let Some(psnr) = row.get("latest_psnr_db").and_then(Value::as_f64) {
        format!("{psnr:.2}")
    } else {
        "n/a".to_owned()
    };
    println!(
        "  {case}: {fps:.2} output fps, total={total:.2} ms, model={model:.2} ms, gaze={:.2}%, psnr={psnr} dB, adapter={adapter}",
        gaze * 100.0
    );
    Ok(())
}

fn check_burn_jepa_sparse_readout_integration(repo: &Path) -> Result<()> {
    let manifest = repo.join("Cargo.toml");
    let bench = repo.join("benches/autogaze_sparse_jepa_pipeline.rs");
    ensure!(
        manifest.is_file() && bench.is_file(),
        "expected burn_jepa checkout with Cargo.toml and benches/autogaze_sparse_jepa_pipeline.rs: {}",
        repo.display()
    );
    let manifest_text = fs::read_to_string(&manifest)?;
    let bench_text = fs::read_to_string(&bench)?;
    let mut errors = Vec::new();
    for (haystack, needle, message) in [
        (
            manifest_text.as_str(),
            "burn_autogaze",
            "burn_jepa should depend on burn_autogaze for sparse readout helpers",
        ),
        (
            bench_text.as_str(),
            "generated_to_frame_readout_tokens",
            "bench should call burn_autogaze::generated_to_frame_readout_tokens for per-frame image readout",
        ),
        (
            bench_text.as_str(),
            "generated_to_video_readout_tokens",
            "bench should call burn_autogaze::generated_to_video_readout_tokens for context mask projection",
        ),
        (
            bench_text.as_str(),
            "SparseReadoutGrid",
            "bench should use burn_autogaze::SparseReadoutGrid for AutoGaze image-token geometry",
        ),
        (
            bench_text.as_str(),
            "SparseVideoReadoutGrid",
            "bench should use burn_autogaze::SparseVideoReadoutGrid for downstream video-token geometry",
        ),
        (
            bench_text.as_str(),
            "SparseVideoReadoutOptions",
            "bench should use burn_autogaze::SparseVideoReadoutOptions for tubelet/exact-token projection",
        ),
    ] {
        if !haystack.contains(needle) {
            errors.push(format!("missing: {message}"));
        }
    }
    for (needle, message) in [
        (
            "fn generated_frame_tokens",
            "local generated_frame_tokens duplicates AutoGaze generated-output decoding",
        ),
        (
            "fn context_mask_from_autogaze_generated",
            "local context_mask_from_autogaze_generated duplicates AutoGaze image/video projection",
        ),
        (
            "raw_token - frame_offset",
            "bench should not manually subtract frame offsets from generated AutoGaze token ids",
        ),
        (
            "gazing_pos.first()",
            "bench should not manually index generated AutoGaze gazing_pos",
        ),
    ] {
        if bench_text.contains(needle) {
            errors.push(format!("still present: {message}"));
        }
    }
    if errors.is_empty() {
        println!(
            "burn_jepa AutoGaze sparse readout integration looks migrated: {}",
            repo.display()
        );
        return Ok(());
    }
    for error in &errors {
        eprintln!("{error}");
    }
    bail!(
        "Expected migration shape:\n  - use burn_autogaze::generated_to_frame_readout_tokens for temporal stream frame tokens\n  - use burn_autogaze::generated_to_video_readout_tokens for SparseTokenMask context indices\n  - keep burn_jepa's SparseTokenMask, target-mask selection, plan caching, and burn_flex_gmm dispatch in burn_jepa"
    )
}

#[derive(Debug, Deserialize)]
struct FixtureMatrix {
    #[serde(default)]
    defaults: FixtureDefaults,
    cases: Vec<FixtureCase>,
}

#[derive(Debug, Default, Deserialize)]
struct FixtureDefaults {
    model_dir: Option<PathBuf>,
    frames: Option<u32>,
    gazing_ratio: Option<f64>,
    task_loss_requirement: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct FixtureCase {
    name: String,
    video: PathBuf,
    out_dir: PathBuf,
    model_dir: Option<PathBuf>,
    frames: Option<u32>,
    gazing_ratio: Option<f64>,
    task_loss_requirement: Option<f64>,
    #[serde(default)]
    extra_args: Vec<String>,
}

fn upstream_fixture_matrix(root: PathBuf, args: UpstreamFixtureMatrixArgs) -> Result<()> {
    let manifest_path = resolve_path(&root, args.manifest);
    let manifest: FixtureMatrix = serde_json::from_str(
        &fs::read_to_string(&manifest_path)
            .with_context(|| format!("read {}", manifest_path.display()))?,
    )
    .with_context(|| format!("parse {}", manifest_path.display()))?;
    ensure!(
        !manifest.cases.is_empty(),
        "manifest must contain at least one case"
    );
    let runner = Runner::new(
        root.clone(),
        &CommonArgs {
            cargo: args.cargo.clone(),
            dry_run: args.dry_run,
        },
        None,
    );
    let selected = args
        .case_names
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let mut seen = std::collections::BTreeSet::new();
    let mut ran_any = false;
    for case in manifest.cases {
        ensure!(
            seen.insert(case.name.clone()),
            "duplicate case name: {}",
            case.name
        );
        if !selected.is_empty() && !selected.contains(&case.name) {
            continue;
        }
        let out_dir = resolve_path(&root, case.out_dir);
        if !args.allow_outside_fixtures {
            ensure_under_fixtures(&root, &out_dir)?;
        }
        if args.skip_existing && out_dir.join("fixture_outputs.safetensors").is_file() {
            println!(
                "skipping {}: fixture_outputs.safetensors already exists",
                case.name
            );
            continue;
        }
        let video = resolve_path(&root, case.video);
        let model_dir = args
            .model_dir
            .clone()
            .or(case.model_dir)
            .or(manifest.defaults.model_dir.clone())
            .map(|path| resolve_path(&root, path));
        if !args.dry_run {
            ensure!(
                video.is_file(),
                "case {}: video not found: {}",
                case.name,
                video.display()
            );
            if let Some(model_dir) = &model_dir {
                ensure!(
                    model_dir.is_dir(),
                    "case {}: model_dir not found: {}",
                    case.name,
                    model_dir.display()
                );
            }
        }
        let mut command_args = vec![
            OsString::from(root.join("xtask/assets/generate_upstream_fixture.py")),
            OsString::from("--video"),
            OsString::from(video),
            OsString::from("--out-dir"),
            OsString::from(out_dir),
        ];
        if let Some(model_dir) = model_dir {
            command_args.extend([OsString::from("--model-dir"), OsString::from(model_dir)]);
        }
        if let Some(frames) = case.frames.or(manifest.defaults.frames) {
            command_args.extend([
                OsString::from("--frames"),
                OsString::from(frames.to_string()),
            ]);
        }
        if let Some(gazing_ratio) = case.gazing_ratio.or(manifest.defaults.gazing_ratio) {
            command_args.extend([
                OsString::from("--gazing-ratio"),
                OsString::from(gazing_ratio.to_string()),
            ]);
        }
        if let Some(task_loss) = case
            .task_loss_requirement
            .or(manifest.defaults.task_loss_requirement)
        {
            command_args.extend([
                OsString::from("--task-loss-requirement"),
                OsString::from(task_loss.to_string()),
            ]);
        }
        command_args.extend(case.extra_args.into_iter().map(OsString::from));
        println!("# {}", case.name);
        runner.run(runner.command("python3", command_args))?;
        ran_any = true;
    }
    if let Some(missing) = selected.difference(&seen).next() {
        bail!("unknown case: {missing}");
    }
    ensure!(ran_any, "no fixture cases selected");
    if args.run_parity_test {
        println!("# fixture-only parity test");
        runner.run(runner.command(
            &args.cargo,
            [
                "test",
                "-p",
                "burn_autogaze",
                "--features",
                "ndarray",
                "--test",
                "native_autogaze_generate_parity",
                "upstream_generated_masks_decode_without_model_snapshot",
                "--",
                "--nocapture",
            ],
        ))?;
    }
    Ok(())
}

fn ensure_under_fixtures(root: &Path, path: &Path) -> Result<()> {
    let fixtures = root.join("tests/fixtures");
    let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let fixtures = fixtures.canonicalize().unwrap_or(fixtures);
    ensure!(
        path.starts_with(&fixtures),
        "fixture out_dir must be under tests/fixtures unless --allow-outside-fixtures is passed: {}",
        path.display()
    );
    Ok(())
}

fn resolve_path(root: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        root.join(path)
    }
}

fn load_json(path: &Path) -> Result<Value> {
    serde_json::from_str(
        &fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?,
    )
    .with_context(|| format!("parse {}", path.display()))
}

fn wasm_bindgen_version(root: &Path) -> Result<String> {
    let lock = fs::read_to_string(root.join("Cargo.lock"))?;
    let mut in_package = false;
    for line in lock.lines() {
        if line == "[[package]]" {
            in_package = false;
        } else if line == "name = \"wasm-bindgen\"" {
            in_package = true;
        } else if in_package && line.starts_with("version = ") {
            return Ok(line
                .trim_start_matches("version = ")
                .trim_matches('"')
                .to_owned());
        }
    }
    bail!("failed to find wasm-bindgen version in Cargo.lock")
}

fn package_dir(root: &Path) -> Result<PathBuf> {
    let manifest = fs::read_to_string(root.join("Cargo.toml"))?;
    let name = package_field(&manifest, "name")?;
    let version = package_field(&manifest, "version")?;
    Ok(root
        .join("target/package")
        .join(format!("{name}-{version}")))
}

fn package_field(manifest: &str, field: &str) -> Result<String> {
    let mut in_package = false;
    for line in manifest.lines() {
        if line == "[package]" {
            in_package = true;
            continue;
        }
        if in_package && line.starts_with('[') {
            break;
        }
        if in_package && line.trim_start().starts_with(&format!("{field} = ")) {
            return Ok(line
                .split_once('=')
                .map(|(_, value)| value.trim().trim_matches('"').to_owned())
                .unwrap_or_default());
        }
    }
    bail!("failed to read package field {field}")
}

fn workspace_root() -> Result<PathBuf> {
    Ok(PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or_else(|| anyhow!("xtask manifest has no parent"))?
        .to_path_buf())
}

fn shell_escape(value: &OsStr) -> String {
    let value = value.to_string_lossy();
    if value.is_empty() {
        return "''".to_owned();
    }
    if value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || "-_./:=+".contains(ch))
    {
        return value.into_owned();
    }
    format!("'{}'", value.replace('\'', "'\\''"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perf_summary_validator_reports_missing_required_field() {
        let mut data = valid_perf_summary();
        data.as_object_mut()
            .expect("summary object")
            .remove("p95_total_ms");

        let err = validate_summary(&data, false).expect_err("missing p95 should fail");

        assert!(
            err.to_string().contains("p95_total_ms"),
            "error should name the missing field: {err:?}"
        );
    }

    #[test]
    fn perf_summary_validator_rejects_inconsistent_percentile_order() {
        let mut data = valid_perf_summary();
        data["p50_total_ms"] = json!(20.0);
        data["p95_total_ms"] = json!(19.0);

        let err = validate_summary(&data, false).expect_err("p95 below p50 should fail");

        assert!(
            err.to_string().contains("p95_total_ms"),
            "error should explain the percentile mismatch: {err:?}"
        );
    }

    #[test]
    fn perf_summary_validator_rejects_infinite_psnr_with_finite_value() {
        let mut data = valid_perf_summary();
        data["latest_psnr_db_infinite"] = json!(true);

        let err = validate_summary(&data, false).expect_err("conflicting PSNR should fail");

        assert!(
            err.to_string().contains("latest_psnr_db"),
            "error should name the conflicting PSNR field: {err:?}"
        );
    }

    #[test]
    fn aggregate_perf_summary_validator_reports_missing_case_count() {
        let data = json!({
            "min_output_fps": 54.0,
            "max_output_fps": 54.0,
            "cases": [valid_perf_summary()]
        });

        let err =
            validate_aggregate_summary(&data, false).expect_err("missing case_count should fail");

        assert!(
            err.to_string().contains("case_count"),
            "error should name the missing aggregate field: {err:?}"
        );
    }

    #[test]
    fn xtask_production_code_avoids_unrecoverable_panics() {
        let source = include_str!("main.rs");
        let production = source
            .split("\n#[cfg(test)]\nmod tests")
            .next()
            .unwrap_or(source);

        for (line_no, line) in production.lines().enumerate() {
            for forbidden in ["panic!", ".unwrap()", ".expect("] {
                assert!(
                    !line.contains(forbidden),
                    "xtask production code should report contextual errors instead of {forbidden} at line {}",
                    line_no + 1
                );
            }
        }
    }

    #[test]
    fn strict_completion_audit_requires_external_evidence_lanes() {
        let args = completion_audit_args(true, None, false);

        let err = validate_completion_audit_args(&args)
            .expect_err("strict audit without burn_jepa should fail");

        assert!(
            err.to_string().contains("--burn-jepa"),
            "error should name the missing external audit lane: {err:?}"
        );

        let args = completion_audit_args(true, Some(PathBuf::from("../burn_jepa")), false);
        let err = validate_completion_audit_args(&args)
            .expect_err("strict audit without hardware perf should fail");

        assert!(
            err.to_string().contains("--hardware-perf"),
            "error should name the missing hardware audit lane: {err:?}"
        );
    }

    #[test]
    fn completion_audit_rejects_zero_frame_hardware_runs() {
        let args = CompletionAuditArgs {
            frames: 0,
            ..completion_audit_args(false, None, false)
        };

        let err = validate_completion_audit_args(&args).expect_err("zero frames should fail");

        assert!(
            err.to_string().contains("--frames"),
            "error should name the invalid frames argument: {err:?}"
        );
    }

    #[test]
    fn completion_audit_rejects_zero_case_timeout() {
        let args = CompletionAuditArgs {
            case_timeout_seconds: 0,
            ..completion_audit_args(false, None, false)
        };

        let err = validate_completion_audit_args(&args).expect_err("zero timeout should fail");

        assert!(
            err.to_string().contains("--case-timeout-seconds"),
            "error should name the invalid timeout argument: {err:?}"
        );
    }

    #[test]
    fn completion_audit_evidence_report_preserves_multiple_failures() {
        let err = finish_completion_audit_evidence(vec![
            "burn_jepa migration audit failed".to_owned(),
            "native hardware Bevy perf audit failed".to_owned(),
        ])
        .expect_err("evidence failures should fail the audit");
        let message = err.to_string();

        assert!(
            message.contains("burn_jepa migration audit failed")
                && message.contains("native hardware Bevy perf audit failed"),
            "completion audit should report every requested evidence failure: {err:?}"
        );
    }

    #[test]
    fn bevy_perf_profile_defaults_to_release_for_throughput_evidence() {
        let args = completion_audit_args(false, None, true);
        assert!(matches!(args.perf_profile, BevyPerfBuildProfile::Release));
        assert_eq!(
            BevyPerfBuildProfile::Release.cargo_run_args(),
            &["--release"]
        );
        assert!(BevyPerfBuildProfile::Dev.cargo_run_args().is_empty());
    }

    #[test]
    fn perf_case_metadata_records_xtask_measurement_context() {
        let mut data = valid_perf_summary();
        augment_perf_case_metadata(
            &mut data,
            "realtime-static-cpu",
            BevyPerfBuildProfile::Release,
            Duration::from_secs(42),
            Path::new("target/autogaze-bevy-perf/cache"),
        )
        .expect("metadata should be inserted");

        assert_eq!(data["case"], "realtime-static-cpu");
        assert_eq!(data["xtask_build_profile"], "release");
        assert_eq!(data["xtask_case_timeout_seconds"], 42);
        assert_eq!(data["xtask_cache_dir"], "target/autogaze-bevy-perf/cache");
        validate_summary(&data, true).expect("augmented summary should validate");
    }

    #[test]
    fn perf_aggregate_excludes_manifest_and_summary_json() {
        assert!(is_perf_case_json_path(Path::new(
            "realtime-static-cpu.json"
        )));
        assert!(!is_perf_case_json_path(Path::new("summary.json")));
        assert!(!is_perf_case_json_path(Path::new(
            BEVY_PERF_MATRIX_MANIFEST
        )));
        assert!(!is_perf_case_json_path(Path::new(
            "realtime-static-cpu.log"
        )));
    }

    #[test]
    fn perf_matrix_manifest_records_measurement_plan() {
        let root = std::env::temp_dir().join(format!(
            "burn-autogaze-xtask-manifest-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let args = BevyPerfMatrixArgs {
            common: CommonArgs {
                cargo: "cargo".to_owned(),
                dry_run: false,
            },
            frames: 7,
            image: PathBuf::from("fixture.png"),
            out: root.clone(),
            camera: true,
            case_timeout_seconds: 42,
            profile: BevyPerfBuildProfile::Release,
        };
        write_perf_matrix_manifest(
            &root,
            &args,
            &[("case-a", vec!["--mode", "realtime"], true)],
        )
        .expect("manifest should write");
        let manifest = load_json(&root.join(BEVY_PERF_MATRIX_MANIFEST)).expect("manifest json");

        assert_eq!(manifest["frames"], 7);
        assert_eq!(manifest["xtask_build_profile"], "release");
        assert_eq!(manifest["xtask_case_timeout_seconds"], 42);
        assert_eq!(manifest["cases"][0]["case"], "case-a");
        assert_eq!(manifest["cases"][0]["include_static_source"], true);
        let _ = fs::remove_dir_all(&root);
    }

    fn completion_audit_args(
        strict: bool,
        burn_jepa: Option<PathBuf>,
        hardware_perf: bool,
    ) -> CompletionAuditArgs {
        CompletionAuditArgs {
            common: CommonArgs {
                cargo: "cargo".to_owned(),
                dry_run: true,
            },
            burn_jepa,
            hardware_perf,
            strict,
            frames: 120,
            case_timeout_seconds: BEVY_PERF_CASE_TIMEOUT_SECS,
            perf_profile: BevyPerfBuildProfile::Release,
            out: PathBuf::from("target/autogaze-bevy-perf-audit"),
        }
    }

    fn valid_perf_summary() -> Value {
        json!({
            "avg_output_fps": 54.0,
            "avg_model_frame_fps": 54.0,
            "avg_input_fps": 60.0,
            "avg_total_ms": 18.0,
            "p50_total_ms": 17.0,
            "p95_total_ms": 19.0,
            "avg_model_ms": 6.0,
            "avg_input_ms": 1.0,
            "avg_pack_ms": 1.0,
            "avg_visualize_ms": 2.0,
            "avg_visualize_cpu_ms": 1.0,
            "avg_tensor_ms": 1.0,
            "avg_display_ms": 1.0,
            "avg_gaze_update_ratio": 0.25,
            "latest_gaze_update_ratio": 0.2,
            "processed_frames": 4,
            "processed_model_frames": 8,
            "latest_clip_frames": 2,
            "latest_model_frames": 2,
            "latest_width": 640,
            "latest_height": 360,
            "latest_trace_points": 12,
            "latest_sequence": 3,
            "target_frames": 4,
            "mode": "resize-224",
            "visualization_mode": "interframe",
            "display_transfer": "gpu",
            "latest_tensor_interframe_path": "sparse-rects",
            "streaming_cache": true,
            "streaming_cache_effective": true,
            "latest_psnr_db": 35.0,
            "latest_psnr_db_infinite": false,
            "ema_psnr_db": 34.0,
            "ema_psnr_db_infinite": false,
            "show_psnr": true,
            "configured_max_in_flight": 1,
            "effective_max_in_flight": 1,
            "frames_per_clip": 2,
            "top_k": 10,
            "tile_batch_size": 64,
            "inference_width": 640,
            "inference_height": 360,
            "max_gaze_tokens_each_frame": 0,
            "tensor_sparse_update_max_rects": 256,
            "tensor_sparse_update_max_ratio": 0.35,
            "render_adapter_device_type": "DiscreteGpu",
            "render_adapter_name": "test gpu",
            "case": "self-test"
        })
    }
}
