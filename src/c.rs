pub mod c_interface {
    pub mod noises {
        #[no_mangle]
        pub extern "C" fn gn(n: usize, t: f64) -> *mut f64 {
            let gn = crate::noises::gn::gn(n, Some(t));
            let gn = gn.into_boxed_slice();
            let gn = Box::into_raw(gn);
            gn as *mut f64
        }

        #[no_mangle]
        pub extern "C" fn fgn_fft(hurst: f64, n: usize, t: f64) -> *mut f64 {
            let fgn = crate::noises::fgn_fft::fgn(hurst, n, t);
            let fgn = fgn.into_boxed_slice();
            let fgn = Box::into_raw(fgn);
            fgn as *mut f64
        }

        #[no_mangle]
        pub extern "C" fn fgn_cholesky(hurst: f64, n: usize, t: f64) -> *mut f64 {
            let fgn = crate::noises::fgn_cholesky::fgn(hurst, n, t);
            let fgn = fgn.into_boxed_slice();
            let fgn = Box::into_raw(fgn);
            fgn as *mut f64
        }
    }

    pub mod processes {
        use crate::utils::NoiseGenerationMethod;

        #[no_mangle]
        pub extern "C" fn bm(n: isize, t: f64) -> *mut f64 {
            let bm = crate::processes::bm::bm(n as usize, Some(t));
            let bm = bm.into_boxed_slice();
            let bm = Box::into_raw(bm);
            bm as *mut f64
        }

        #[no_mangle]
        pub extern "C" fn fbm(
            hurst: f64,
            n: usize,
            t: f64,
            method: NoiseGenerationMethod,
        ) -> *mut f64 {
            let fbm = crate::processes::fbm::fbm(hurst, n, Some(t), Some(method));
            let fbm = fbm.into_boxed_slice();
            let fbm = Box::into_raw(fbm);
            fbm as *mut f64
        }
    }

    pub mod diffusions {
        #[no_mangle]
        pub extern "C" fn ou(
            n: isize,
            mu: f64,
            sigma: f64,
            theta: f64,
            x0: f64,
            t: f64,
        ) -> *mut f64 {
            let ou = crate::diffusions::ou::ou(mu, sigma, theta, n as usize, Some(x0), Some(t));
            let ou = ou.into_boxed_slice();
            let ou = Box::into_raw(ou);
            ou as *mut f64
        }

        #[no_mangle]
        pub extern "C" fn fou(
            hurst: f64,
            mu: f64,
            sigma: f64,
            theta: f64,
            n: isize,
            x0: f64,
            t: f64,
            method: crate::utils::NoiseGenerationMethod,
        ) -> *mut f64 {
            let fou = crate::diffusions::ou::fou(
                hurst,
                mu,
                sigma,
                theta,
                n as usize,
                Some(x0),
                Some(t),
                Some(method),
            );
            let fou = fou.into_boxed_slice();
            let fou = Box::into_raw(fou);
            fou as *mut f64
        }

        #[no_mangle]
        pub extern "C" fn cir(
            theta: f64,
            mu: f64,
            sigma: f64,
            n: usize,
            x0: f64,
            t: f64,
            use_sym: bool,
        ) -> *mut f64 {
            let cir =
                crate::diffusions::cir::cir(theta, mu, sigma, n, Some(x0), Some(t), Some(use_sym));
            let cir = cir.into_boxed_slice();
            let cir = Box::into_raw(cir);
            cir as *mut f64
        }

        #[no_mangle]
        pub extern "C" fn fcir(
            hurst: f64,
            theta: f64,
            mu: f64,
            sigma: f64,
            n: usize,
            x0: f64,
            t: f64,
            method: crate::utils::NoiseGenerationMethod,
            use_sym: bool,
        ) -> *mut f64 {
            let fcir = crate::diffusions::cir::fcir(
                hurst,
                theta,
                mu,
                sigma,
                n,
                Some(x0),
                Some(t),
                Some(method),
                Some(use_sym),
            );
            let fcir = fcir.into_boxed_slice();
            let fcir = Box::into_raw(fcir);
            fcir as *mut f64
        }

        #[no_mangle]
        pub extern "C" fn gbm(mu: f64, sigma: f64, n: usize, x0: f64, t: f64) -> *mut f64 {
            let gbm = crate::diffusions::gbm::gbm(mu, sigma, n, Some(x0), Some(t));
            let gbm = gbm.into_boxed_slice();
            let gbm = Box::into_raw(gbm);
            gbm as *mut f64
        }

        #[no_mangle]
        pub extern "C" fn fgbm(
            hurst: f64,
            mu: f64,
            sigma: f64,
            x0: f64,
            n: usize,
            t: f64,
            method: crate::utils::NoiseGenerationMethod,
        ) -> *mut f64 {
            let fgbm =
                crate::diffusions::gbm::fgbm(hurst, mu, sigma, n, Some(x0), Some(t), Some(method));
            let fgbm = fgbm.into_boxed_slice();
            let fgbm = Box::into_raw(fgbm);
            fgbm as *mut f64
        }

        #[no_mangle]
        pub extern "C" fn jacobi(
            alpha: f64,
            beta: f64,
            sigma: f64,
            n: usize,
            x0: f64,
            t: f64,
        ) -> *mut f64 {
            let jacobi =
                crate::diffusions::jacobi::jacobi(alpha, beta, sigma, n, Some(x0), Some(t));
            let jacobi = jacobi.into_boxed_slice();
            let jacobi = Box::into_raw(jacobi);
            jacobi as *mut f64
        }

        #[no_mangle]
        pub extern "C" fn fjacobi(
            hurst: f64,
            alpha: f64,
            beta: f64,
            sigma: f64,
            n: usize,
            x0: f64,
            t: f64,
            method: crate::utils::NoiseGenerationMethod,
        ) -> *mut f64 {
            let fjacobi = crate::diffusions::jacobi::fjacobi(
                hurst,
                alpha,
                beta,
                sigma,
                n,
                Some(x0),
                Some(t),
                Some(method),
            );
            let fjacobi = fjacobi.into_boxed_slice();
            let fjacobi = Box::into_raw(fjacobi);
            fjacobi as *mut f64
        }
    }
}
