use ndarray::{concatenate, prelude::*};
use ndrustfft::{ndfft, FftHandler};
use num_complex::Complex;
use rand::Rng;
use rand_distr::StandardNormal;

// TODO: improve performance
pub fn fgn(hurst: f64, n: usize, t: Option<f64>) -> Vec<f64> {
    if !(0.0..=1.0).contains(&hurst) {
        panic!("Hurst parameter must be between 0 and 1");
    }

    let r = Array::from_shape_fn((n + 1,), |i| {
        if i == 0 {
            1.0
        } else {
            0.5 * ((i as f64 + 1.0).powf(2.0 * hurst) - 2.0 * (i as f64).powf(2.0 * hurst)
                + (i as f64 - 1.0).powf(2.0 * hurst))
        }
    });

    let r = concatenate(
        Axis(0),
        #[allow(clippy::reversed_empty_ranges)]
        &[r.view(), r.slice(s![..;-1]).slice(s![1..-1]).view()],
    )
    .unwrap();

    let mut data = Array1::<Complex<f64>>::zeros(r.len());
    for (i, v) in r.iter().enumerate() {
        data[i] = Complex::new(*v, 0.0);
    }
    let mut r_fft = FftHandler::new(r.len());
    let mut lambda = Array1::<Complex<f64>>::zeros(r.len());
    ndfft(&data, &mut lambda, &mut r_fft, 0);
    let lambda = lambda.mapv(|x| x.re / (2.0 * n as f64)).mapv(f64::sqrt);

    let mut rng = rand::thread_rng();
    let mut rnd = Array1::<Complex<f64>>::zeros(2 * n);
    for (_, v) in rnd.iter_mut().enumerate() {
        let real: f64 = rng.sample(StandardNormal);
        let imag: f64 = rng.sample(StandardNormal);
        *v = Complex::new(real, imag);
    }

    let mut fgn = Array1::<Complex<f64>>::zeros(2 * n);
    for (i, v) in rnd.iter().enumerate() {
        fgn[i] = lambda[i] * v;
    }
    let mut fgn_fft_handler = FftHandler::new(2 * n);
    let mut fgn_fft = Array1::<Complex<f64>>::zeros(2 * n);
    ndfft(&fgn, &mut fgn_fft, &mut fgn_fft_handler, 0);

    let fgn = fgn_fft.slice(s![1..n + 1]).mapv(|x| x.re);
    fgn.mapv(|x| (x * (n as f64).powf(-hurst)) * t.unwrap_or(1.0).powf(hurst))
        .to_vec()
}
