use std::path::Path;

fn bevy_source() -> Option<String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("crates")
        .join("bevy_burn_autogaze")
        .join("src")
        .join("lib.rs");
    match std::fs::read_to_string(&path) {
        Ok(source) => Some(source),
        Err(err) => {
            eprintln!(
                "skipping Bevy source hygiene check: failed to read {}: {err}",
                path.display()
            );
            None
        }
    }
}

fn function_body_after<'a>(source: &'a str, marker: &str, fn_name: &str) -> &'a str {
    let marker_start = source.find(marker).expect("marker should exist");
    let search = &source[marker_start..];
    let fn_start = search.find(fn_name).expect("function should exist");
    let body_search = &search[fn_start..];
    let body_start = body_search.find('{').expect("function body should start");
    let absolute_body_start = marker_start + fn_start + body_start;
    let mut depth = 0usize;
    for (offset, ch) in source[absolute_body_start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return &source[absolute_body_start..=absolute_body_start + offset];
                }
            }
            _ => {}
        }
    }
    panic!("function body should close");
}

fn line_has_guard(source: &str, needle: &str, guard: &str) -> bool {
    let lines: Vec<_> = source.lines().collect();
    lines.iter().enumerate().all(|(index, line)| {
        if !line.contains(needle) {
            return true;
        }
        index > 0 && lines[index - 1].trim() == guard
    })
}

fn optional_source(path: impl AsRef<Path>) -> Option<String> {
    let path = path.as_ref();
    match std::fs::read_to_string(path) {
        Ok(source) => Some(source),
        Err(err) => {
            eprintln!(
                "skipping source hygiene check: failed to read {}: {err}",
                path.display()
            );
            None
        }
    }
}

fn source_before_tests(source: &str) -> &str {
    source
        .split("\n#[cfg(test)]\nmod tests")
        .next()
        .unwrap_or(source)
}

fn source_before_first_test_module(source: &str) -> String {
    let lines = source.lines().collect::<Vec<_>>();
    for (idx, line) in lines.iter().enumerate() {
        if !line.trim_start().starts_with("mod tests") {
            continue;
        }

        let is_test_cfg = |line: &str| {
            let line = line.trim();
            line.starts_with("#[cfg(") && line.contains("test")
        };
        let cfg_idx = if idx > 0 && is_test_cfg(lines[idx - 1]) {
            Some(idx - 1)
        } else if idx > 1 && lines[idx - 1].trim().is_empty() && is_test_cfg(lines[idx - 2]) {
            Some(idx - 2)
        } else {
            None
        };

        if let Some(cutoff) = cfg_idx {
            return lines[..cutoff].join("\n");
        }
    }

    source.to_string()
}

#[test]
fn bevy_readout_uses_async_tensor_data_path() {
    let Some(source) = bevy_source() else {
        return;
    };
    let body = function_body_after(
        &source,
        "async fn run_autogaze_readout",
        "async fn run_autogaze_readout",
    );
    assert!(body.contains("readout_prepared_run_async"));
    assert!(body.contains("into_data_async") || body.contains("readout_prepared_run_async"));
    assert!(
        !body.contains("readout_prepared_run(trace_input"),
        "Bevy readout must not call the synchronous prepared readout API"
    );
}

#[test]
fn bevy_wasm_timing_does_not_use_std_instant() {
    let Some(source) = bevy_source() else {
        return;
    };
    assert!(
        line_has_guard(
            &source,
            "use std::time::Instant",
            "#[cfg(not(target_arch = \"wasm32\"))]"
        ),
        "std::time::Instant import must stay native-only"
    );
    assert!(
        line_has_guard(
            &source,
            "type Timestamp = Instant",
            "#[cfg(not(target_arch = \"wasm32\"))]"
        ),
        "Instant-backed timestamp type must stay native-only"
    );
    let wasm_timestamp = function_body_after(
        &source,
        "#[cfg(target_arch = \"wasm32\")]\nfn timestamp_now",
        "fn timestamp_now",
    );
    assert!(wasm_timestamp.contains("js_sys::Date::now()"));
    assert!(!wasm_timestamp.contains("Instant::now()"));
}

#[test]
fn bevy_realtime_default_uses_model_generation_budget() {
    let Some(source) = bevy_source() else {
        return;
    };

    assert!(
        source.contains(
            "pub const DEFAULT_REALTIME_MAX_GAZE_TOKENS: usize = DEFAULT_MODEL_GENERATION_BUDGET;"
        ),
        "Bevy realtime defaults should not silently cap the upstream model budget"
    );
}

#[test]
fn public_wasm_api_uses_async_trace_and_no_sync_tensor_readback() {
    let source = include_str!("../src/wasm.rs");
    assert!(source.contains("trace_rgba_clip_with_mode_async"));
    assert!(!source.contains(".into_data()"));
    assert!(!source.contains("Instant::now"));
    assert!(!source.contains("std::time::Instant"));
}

#[test]
fn model_async_greedy_selection_uses_async_readback_only() {
    let source = include_str!("../src/model.rs");
    let body = function_body_after(
        source,
        "async fn greedy_select_multi_tokens_async",
        "async fn greedy_select_multi_tokens_async",
    );
    assert!(body.contains("pack_greedy_logits_task"));
    assert!(body.contains(".into_data_async()"));
    assert!(
        !body.contains(".into_data()"),
        "async greedy selection must not synchronously read tensor data"
    );
}

#[test]
fn greedy_selection_reads_logits_once_per_decoder_step() {
    let source = include_str!("../src/model.rs");
    assert!(
        !source.contains("fn greedy_step_tensor"),
        "greedy selection should not launch/read back one tensor per multi-token slot"
    );
    assert!(
        source.contains("fn pack_greedy_logits_task"),
        "greedy selection should pack logits and task loss for one readback per decoder step"
    );
}

#[test]
fn production_pipeline_surfaces_avoid_unrecoverable_panics() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let checked_sources = [
        manifest.join("src").join("config.rs"),
        manifest.join("src").join("lib.rs"),
        manifest.join("src").join("metrics.rs"),
        manifest.join("src").join("model.rs"),
        manifest.join("src").join("nodes.rs"),
        manifest.join("src").join("pipeline.rs"),
        manifest.join("src").join("pyramid.rs"),
        manifest.join("src").join("readout.rs"),
        manifest.join("src").join("runtime.rs"),
        manifest.join("src").join("safetensors_io.rs"),
        manifest.join("src").join("trace.rs"),
        manifest.join("src").join("visualization.rs"),
        manifest.join("src").join("wasm.rs"),
        manifest
            .join("crates")
            .join("bevy_burn_autogaze")
            .join("src")
            .join("lib.rs"),
        manifest
            .join("crates")
            .join("bevy_burn_autogaze")
            .join("src")
            .join("main.rs"),
        manifest
            .join("crates")
            .join("bevy_burn_autogaze")
            .join("src")
            .join("platform.rs"),
    ];

    for path in checked_sources {
        let Some(source) = optional_source(&path) else {
            continue;
        };
        let production = source_before_first_test_module(&source);
        for (line_no, line) in production.lines().enumerate() {
            for forbidden in ["panic!", ".unwrap()", ".expect("] {
                assert!(
                    !line.contains(forbidden),
                    "{}:{} production code should propagate errors or return fallbacks instead of {forbidden}",
                    path.display(),
                    line_no + 1
                );
            }
        }
    }
}

#[test]
fn bevy_visualization_delegates_to_core_visualization_helpers() {
    let Some(source) = bevy_source() else {
        return;
    };
    let production = source_before_tests(&source);
    let tensor_body = function_body_after(
        production,
        "fn visualize_rgba_tensor",
        "fn visualize_rgba_tensor",
    );
    assert!(tensor_body.contains(".gpu"));
    assert!(tensor_body.contains("visualize_normalized_rgb_clip_panels"));
    assert!(
        !tensor_body.contains("copy_sparse_update_rgba")
            && !tensor_body.contains("fixation_sparse_update_plan")
            && !tensor_body.contains("dense_interframe_update_tensor"),
        "Bevy GPU visualization should delegate mask/interframe composition to burn_autogaze"
    );

    let byte_body = function_body_after(
        production,
        "fn visualize_rgba_bytes",
        "fn visualize_rgba_bytes",
    );
    assert!(byte_body.contains(".cpu"));
    assert!(byte_body.contains("visualize_rgba("));
    assert!(byte_body.contains("visualize_rgba_panels("));
    assert!(byte_body.contains("output_psnr_db(rgba)"));
    assert!(
        !byte_body.contains("rgba_psnr_db")
            && !byte_body.contains("copy_sparse_update_rgba")
            && !byte_body.contains("fixation_sparse_update_plan")
            && !byte_body.contains("dense_interframe_update_tensor"),
        "Bevy CPU visualization should delegate PSNR, mask, and interframe math to burn_autogaze"
    );

    for forbidden in [
        "fn rgba_psnr_db",
        "fn fixation_sparse_update_plan",
        "fn copy_sparse_update_rgba",
        "fn dense_interframe_update_tensor",
        "generated_scale_token_masks",
        "generated_frame_fixations_from_layouts",
    ] {
        assert!(
            !production.contains(forbidden),
            "Bevy production source should not duplicate core visualization/model logic; found {forbidden}"
        );
    }
}

#[test]
fn bevy_gpu_display_path_is_tensor_backed_bevy_burn_interop() {
    let Some(source) = bevy_source() else {
        return;
    };
    let production = source_before_tests(&source);

    assert!(
        production.contains("BevyBurnBridgePlugin::<AutoGazeBevyBackend>::default()"),
        "Bevy app should initialize the bevy_burn bridge so Burn tensors use Bevy's render device"
    );

    let panel_upload = function_body_after(
        production,
        "fn set_gpu_panel_upload_handle",
        "fn set_gpu_panel_upload_handle",
    );
    for expected in [
        "BevyBurnHandle::<AutoGazeBevyBackend>",
        "BindingDirection::BurnToBevy",
        "TransferKind::Gpu",
    ] {
        assert!(
            panel_upload.contains(expected),
            "GPU display upload should be tensor-backed through bevy_burn; missing {expected}"
        );
    }

    let tensor_body = function_body_after(
        production,
        "fn visualize_rgba_tensor",
        "fn visualize_rgba_tensor",
    );
    assert!(
        tensor_body.contains("output_rgba_bytes: 0"),
        "GPU visualization should not emit host RGBA for display"
    );
    assert!(
        tensor_body.contains("output_tensor_bytes"),
        "GPU visualization should report tensor-resident output bytes"
    );
}

#[test]
fn bevy_runtime_backend_is_webgpu_and_avoids_readback_input_bridge() {
    let Some(source) = bevy_source() else {
        return;
    };
    let production = source_before_tests(&source);

    assert!(
        production.contains("pub type AutoGazeBevyBackend = burn::backend::WebGpu<f32, i32>;"),
        "bevy_burn_autogaze should run the model on Burn WebGPU, not a CPU backend"
    );
    assert!(
        production.contains("const AUTO_GAZE_BEVY_BACKEND_NAME: &str = \"webgpu\";"),
        "perf summaries should expose the Bevy Burn backend explicitly"
    );
    assert!(
        !production.contains("NdArray"),
        "Bevy production source should not instantiate the ndarray CPU backend"
    );
    assert!(
        !production.contains("BindingDirection::BevyToBurn"),
        "the current bevy_burn BevyToBurn path stages through a CPU readback; Bevy input should use the core RGBA-to-tensor upload path until a true texture-to-tensor GPU path exists"
    );
}

#[test]
fn bevy_gpu_display_input_reuses_prepared_model_tensor_when_valid() {
    let Some(source) = bevy_source() else {
        return;
    };
    let production = source_before_tests(&source);

    let helper = function_body_after(
        production,
        "fn display_tensor_from_prepared_trace",
        "fn display_tensor_from_prepared_trace",
    );
    for expected in [
        "trace_input.video.shape().dims::<5>()",
        "dims[3] == clip.height()",
        "dims[4] == clip.width()",
        "video_frame_tensor(trace_input.video.clone(), trace_input.frame_index)",
        "DisplayInputResidency::ModelTensorReuse",
        "display_tensor_from_clip(clip, options, device)?",
        "DisplayInputResidency::HostRgbaUpload",
    ] {
        assert!(
            helper.contains(expected),
            "display input preparation should reuse full-resolution model tensors when possible and report host-upload fallbacks; missing {expected}"
        );
    }

    let clip_fallback = function_body_after(
        production,
        "fn display_tensor_from_clip",
        "fn display_tensor_from_clip",
    );
    assert!(clip_fallback.contains("clip.last_frame_rgba()?"));
    assert!(clip_fallback.contains("rgba_clip_to_tensor::<AutoGazeBevyBackend>"));
}

#[test]
fn bevy_gpu_display_upload_flag_is_one_shot() {
    let Some(source) = bevy_source() else {
        return;
    };
    let production = source_before_tests(&source);

    let clear = function_body_after(
        production,
        "fn clear_completed_gpu_uploads",
        "fn clear_completed_gpu_uploads",
    );
    assert!(
        clear.contains("handle.upload = false"),
        "completed Burn-to-Bevy uploads should be marked idle so unchanged tensors are not copied every render frame"
    );

    let set = function_body_after(
        production,
        "fn set_gpu_panel_upload_handle",
        "fn set_gpu_panel_upload_handle",
    );
    assert!(set.contains("handle.upload = true"));
    assert!(set.contains("entity.insert(OneShotGpuUpload)"));
}

#[test]
fn bevy_ui_text_uses_reserved_rows_not_image_overlays() {
    let Some(source) = bevy_source() else {
        return;
    };
    let production = source_before_tests(&source);

    let setup = function_body_after(production, "fn setup_ui", "fn setup_ui");
    assert!(
        setup.contains("GridTrack::px(PANEL_LABEL_ROW_HEIGHT)")
            && setup.contains("grid_row: GridPlacement::start(2)")
            && setup.contains("Text(label.to_string())")
            && setup.contains("padding: UiRect::horizontal"),
        "panel labels should live in a reserved grid row above the image textures"
    );
    assert!(
        !setup.contains(
            "position_type: PositionType::Absolute,\n                    top: Val::Px(10.0)"
        ),
        "panel labels must not be absolutely overlaid on the image textures"
    );

    let fit = function_body_after(
        production,
        "fn fit_visualization_node",
        "fn fit_visualization_node",
    );
    assert!(
        fit.contains("metric_panel_top_reserved_height(&config)")
            && fit.contains("available_image_height")
            && fit.contains("PANEL_LABEL_ROW_HEIGHT")
            && fit.contains("node.top = Val::Px(reserved_top"),
        "visualization fitting should reserve vertical UI space above the image rows"
    );

    for metric_setup in [
        ("fn fps_display_setup", "metric_overlay_top(0)"),
        (
            "fn gaze_ratio_display_setup",
            "metric_overlay_top(usize::from(config.show_fps))",
        ),
        ("fn psnr_display_setup", "metric_overlay_top(row)"),
    ] {
        let body = function_body_after(production, metric_setup.0, metric_setup.0);
        assert!(body.contains(metric_setup.1));
        assert!(
            !body.contains("bottom: Val::Px"),
            "{} should use the reserved top metric band, not bottom image overlay",
            metric_setup.0
        );
    }
}

#[test]
fn bevy_metrics_delegate_to_core_metric_helpers() {
    let Some(source) = bevy_source() else {
        return;
    };
    let production = source_before_tests(&source);
    assert!(production.contains("struct GazeRatioStats(AutoGazeGazeRatioStats);"));
    assert!(production.contains("struct PsnrStats(AutoGazePsnrStats);"));

    let gaze_stats = function_body_after(production, "impl GazeRatioStats", "fn record");
    assert!(gaze_stats.contains("self.0.record(ratio)"));
    let psnr_stats = function_body_after(production, "impl PsnrStats", "fn record");
    assert!(psnr_stats.contains("self.0.record(psnr_db)"));

    let fps_text = function_body_after(production, "fn fps_update_system", "fn fps_update_system");
    assert!(fps_text.contains("format_fps(timing.e2e_fps())"));
    assert!(fps_text.contains("format_fps(value)"));

    let gaze_text = function_body_after(
        production,
        "fn stable_gaze_ratio_text",
        "fn stable_gaze_ratio_text",
    );
    assert!(gaze_text.contains("format_gaze_ratio_percent"));

    let psnr_text = function_body_after(production, "fn stable_psnr_text", "fn stable_psnr_text");
    assert!(psnr_text.contains("format_psnr_db"));

    for forbidden in [
        "sanitize_gaze_ratio",
        "DEFAULT_METRIC_EMA_ALPHA",
        "fn ema_metric",
        "format!(\"{:.1}%\"",
    ] {
        assert!(
            !production.contains(forbidden),
            "Bevy production source should use core metric helpers instead of local metric math; found {forbidden}"
        );
    }
}

#[test]
fn generated_output_decoding_stays_in_core_model_and_readout_helpers() {
    let model = include_str!("../src/model.rs");
    assert!(model.contains("fn generated_frame_fixations_from_layouts"));
    assert!(model.contains("fn generated_scale_token_masks"));

    let readout = include_str!("../src/readout.rs");
    assert!(readout.contains("generated_frame_fixations(generated"));
    let legacy_test = function_body_after(
        readout,
        "fn legacy_burn_jepa_generated_frame_tokens",
        "fn legacy_burn_jepa_generated_frame_tokens",
    );
    let readout_without_legacy_test = readout.replace(legacy_test, "");
    for needle in [
        "generated.gazing_pos.first()",
        "generated.if_padded_gazing.first()",
        "raw_token - frame_offset",
    ] {
        assert!(
            !readout_without_legacy_test.contains(needle),
            "production readout helpers should delegate generated-token decoding to src/model.rs; found {needle}"
        );
    }

    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let checked_sources = [
        manifest.join("benches").join("backend_pipeline.rs"),
        manifest
            .join("examples")
            .join("sparse_video_readout_adapter.rs"),
        manifest.join("examples").join("render_readme_assets.rs"),
        manifest
            .join("crates")
            .join("bevy_burn_autogaze")
            .join("src")
            .join("lib.rs"),
        manifest
            .join("crates")
            .join("bevy_burn_autogaze")
            .join("src")
            .join("main.rs"),
        manifest
            .join("crates")
            .join("bevy_burn_autogaze")
            .join("src")
            .join("platform.rs"),
    ];
    for path in checked_sources {
        let Some(source) = optional_source(&path) else {
            continue;
        };
        for needle in [
            "generated_frame_tokens",
            "context_mask_from_autogaze_generated",
            "generated.gazing_pos.first()",
            "generated.if_padded_gazing.first()",
            "raw_token - frame_offset",
        ] {
            assert!(
                !source.contains(needle),
                "{} must use burn_autogaze readout helpers instead of local generated-output decoding; found {needle}",
                path.display()
            );
        }
    }
}

#[test]
fn repo_tooling_entrypoints_live_in_xtask() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let xtask = manifest.join("xtask").join("src").join("main.rs");
    if !xtask.is_file() {
        eprintln!("skipping repo tooling hygiene check outside workspace checkout");
        return;
    }

    let legacy_tools = manifest.join("tools");
    assert!(
        !legacy_tools.exists(),
        "repo-level command entry points should live in xtask, not tools/*.sh or tools/*.py"
    );

    let checked_docs = [
        manifest.join("README.md"),
        manifest.join("docs").join("completion-audit.md"),
        manifest.join("docs").join("sparse-readout-integration.md"),
        manifest
            .join("crates")
            .join("bevy_burn_autogaze")
            .join("README.md"),
        manifest.join("web").join("README.md"),
        manifest.join(".github").join("workflows").join("test.yml"),
        manifest
            .join(".github")
            .join("workflows")
            .join("deploy-pages.yml"),
    ];
    for path in checked_docs {
        let Some(source) = optional_source(&path) else {
            continue;
        };
        assert!(
            !source.contains("tools/check_")
                && !source.contains("tools/run_")
                && !source.contains("tools/validate_")
                && !source.contains("tools/generate_")
                && !source.contains("tools/common.sh")
                && !source.contains("bash -n cargo run")
                && !source.contains("py_compile tools"),
            "{} should use cargo run -p xtask instead of legacy tools scripts",
            path.display()
        );
    }
}

#[test]
fn bevy_platform_selection_is_target_cfg_not_feature_flags() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let bevy_manifest = manifest
        .join("crates")
        .join("bevy_burn_autogaze")
        .join("Cargo.toml");
    let Some(source) = optional_source(&bevy_manifest) else {
        return;
    };

    let features_start = source
        .find("[features]")
        .expect("features section should exist");
    let features_section = source[features_start..]
        .split("\n[dependencies]")
        .next()
        .unwrap_or(&source[features_start..]);
    assert!(
        features_section.contains("default = []"),
        "bevy_burn_autogaze should not require a platform feature for default native runs"
    );
    for forbidden in ["native =", "web =", "wasm ="] {
        assert!(
            !features_section.contains(forbidden),
            "bevy_burn_autogaze should select native/wasm dependencies from target cfg, not a {forbidden} feature"
        );
    }
    assert!(
        source.contains("[target.'cfg(not(target_arch = \"wasm32\"))'.dependencies]")
            && source.contains("[target.'cfg(target_arch = \"wasm32\")'.dependencies]"),
        "bevy_burn_autogaze should keep platform-specific dependencies behind target cfg"
    );

    let checked_docs = [
        manifest.join("README.md"),
        manifest
            .join("crates")
            .join("bevy_burn_autogaze")
            .join("README.md"),
        manifest.join("web").join("README.md"),
        manifest.join("docs").join("completion-audit.md"),
    ];
    for path in checked_docs {
        let Some(source) = optional_source(&path) else {
            continue;
        };
        for forbidden in ["--features native", "features native"] {
            assert!(
                !source.contains(forbidden),
                "{} should document target-based Bevy platform selection instead of {forbidden}",
                path.display()
            );
        }
        for line in source.lines() {
            assert!(
                !line.contains("--features web ") && !line.trim_end().ends_with("--features web"),
                "{} should not document an exact Bevy --features web platform flag",
                path.display()
            );
        }
    }
}

#[test]
fn bevy_realtime_admission_uses_configured_core_policy() {
    let Some(source) = bevy_source() else {
        return;
    };
    assert!(
        !source.contains("const REALTIME_POLICY"),
        "Bevy must not hide realtime admission behind a fixed local policy constant"
    );
    assert!(
        source.contains("pub max_in_flight: usize"),
        "Bevy config should expose the core realtime admission limit"
    );
    assert!(
        source.contains("realtime_policy_from_config"),
        "Bevy should derive realtime admission from the sanitized viewer config"
    );
    let helper = function_body_after(
        &source,
        "pub const fn realtime_policy_from_config",
        "pub const fn realtime_policy_from_config",
    );
    assert!(
        helper.contains("should_use_streaming_cache"),
        "streaming KV-cache mode must cap concurrent inference to preserve cache order"
    );
    assert!(
        helper.contains("DEFAULT_MAX_IN_FLIGHT"),
        "streaming KV-cache mode should use the shared core in-flight default"
    );
}
