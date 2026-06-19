; Automatic Differentiation Example 5: Gradient Descent Optimization
;
; Fits the polynomial f(x) = c0 + c1·x + c2·x² to four data points
; by minimizing the sum of squared residuals.  The optimization loop
; runs entirely in Remora — no Python orchestration needed.
;
; (grad poly-loss) is resolved once at compile time.  The fold calls
; the resulting gradient function 200 times, each time with the
; current parameter values.
;
; Run (interpreter):
;   remorac --syntax lisp --target interp examples/ad_optimize.lisp
;
; Run (compiled CPU):
;   remorac --syntax lisp --target cpu examples/ad_optimize.lisp
;
; Expected output (both paths):
;   [0.512337, 0.433115, 0.911621]
;
; This means the fitted polynomial is approximately:
;   f(x) ≈ 0.51 + 0.43x + 0.91x²

; ── The polynomial being fitted ──────────────────────────────────────

(define/pi ()
  (poly-eval [coeffs (Array Float 3) x Float] Float)
  (let* ((c0 (index-item coeffs 0))
         (c1 (index-item coeffs 1))
         (c2 (index-item coeffs 2)))
    (+ c0 (+ (* c1 x) (* c2 (* x x))))))

; ── Loss: sum of squared residuals at four data points ───────────────
;
;   x:  0    1    2    3
;   y:  1    2    5   10

(define/pi ()
  (poly-loss [coeffs (Array Float 3)] Float)
  (let* ((r0 (- (poly-eval coeffs 0.0) 1.0))
         (r1 (- (poly-eval coeffs 1.0) 2.0))
         (r2 (- (poly-eval coeffs 2.0) 5.0))
         (r3 (- (poly-eval coeffs 3.0) 10.0)))
    (+ (* r0 r0) (+ (* r1 r1) (+ (* r2 r2) (* r3 r3))))))

; ── Gradient descent: 200 steps, learning rate 0.001 ─────────────────

(fold (lambda (params step)
        (- params (* 0.001 ((grad poly-loss) params))))
      [0.0 0.0 0.0]
      (iota 200))
