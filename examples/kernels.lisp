; Statistical kernel functions
; Translated from /e/projects/Remora/remora/examples/kernels.rkt
;
; Includes: mean, variance, covariance, autocovariance.
; Excluded: correlation and autocorrelation (require sqrt, not exposed
;   as a user-callable built-in in remorac), sinusoid/Goertzel
;   (require cos, pi, complex numbers), butterfly (requires even?/odd?),
;   matrix multiply m* (uses rank-polymorphic patterns beyond current support).
;
; Run: remorac --syntax lisp --target interp examples/kernels.lisp

; Mean
(define (mean samples)
  (/ (reduce + 0 samples) (length samples)))

; Variance
(define (variance samples)
  (mean (* (- samples (mean samples))
           (- samples (mean samples)))))

; Covariance
(define (covariance xs ys)
  (mean (* (- xs (mean xs))
           (- ys (mean ys)))))

; Autocovariance -- covariance of a signal and a delayed version of itself
; Pads the delayed signal by rotating from the beginning.
(define (autocovariance samples delay)
  (covariance samples (rotate samples delay)))

; --- Examples ---
(mean [1.0 2.0 3.0 4.0 5.0])
(variance [1.0 2.0 3.0 4.0 5.0])
(covariance [1.0 2.0 3.0 4.0 5.0] [5.0 4.0 3.0 2.0 1.0])
(autocovariance [1.0 2.0 3.0 4.0 5.0] 2)
