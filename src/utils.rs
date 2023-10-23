#[repr(C)]
#[derive(Clone, Copy)]
pub enum NoiseGenerationMethod {
    Cholesky,
    Fft,
}
