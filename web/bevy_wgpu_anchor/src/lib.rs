pub fn dependency_anchor() -> (&'static str, &'static str) {
    (
        core::any::type_name::<bevy::prelude::App>(),
        core::any::type_name::<burn_autogaze::WasmAutoGaze>(),
    )
}
