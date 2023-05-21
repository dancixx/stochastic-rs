use rand::Rng;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

fn fbm(hurst: f64, n: usize) {
    // let mut r = vec![0.0; n + 1];
    // r[0] = 1.0;
    // let n_range: Vec<usize> = (1..=n).collect();
    // for i in 1..=n {
    //     r[i] = 0.5
    //         * ((n_range[i - 1] + 1).powf(2.0 * hurst) - 2.0 * n_range[i - 1].powf(2.0 * hurst)
    //             + (n_range[i - 1] - 1).powf(2.0 * hurst));
    // }
    // r.extend(r.iter().rev().skip(1));

    // let fft = FftPlanner::new(false).plan_fft(r.len());
    // let mut lmbd: Vec<f64> = vec![0.0; r.len()];
    // let mut c_r: Vec<Complex<f64>> = r.into_iter().map(Complex::from).collect();
    // fft.process(&mut c_r, &mut lmbd);
    // lmbd = lmbd.iter().map(|x| x.re / (2.0 * n as f64)).collect();

    // let sqrt: Vec<f64> = lmbd.iter().map(|&x| x.sqrt()).collect();

    // let mut rng = rand::thread_rng();
    // let rnd: Vec<Complex<f64>> = (0..(2 * n))
    //     .map(|_| {
    //         let real = rng.gen::<f64>();
    //         let imag = rng.gen::<f64>();
    //         Complex::new(real, imag)
    //     })
    //     .collect();

    // let fft = FftPlanner::new(true).plan_fft(r.len());
    // let mut w: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];
    // fft.process(
    //     &mut (sqrt
    //         .into_iter()
    //         .zip(rnd.into_iter())
    //         .map(|(x, y)| x * y)
    //         .collect()),
    //     &mut w,
    // );
    // let w_real: Vec<f64> = w.into_iter().skip(1).map(|x| x.re).collect();

    // let mut fbm: Vec<f64> = vec![0.0; n + 1];
    // let n_pow_hurst = (n as f64).powf(-hurst);
    // for i in 1..=n {
    //     fbm[i] = n_pow_hurst * w_real[..i].iter().sum::<f64>();
    // }

    // fbm
}
