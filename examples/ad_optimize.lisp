; Automatic Differentiation Example 5: Gradient Descent Optimization
;
; A complete optimization loop in pure Remora — no Python needed.
;
; We reuse the polynomial loss from ad_polynomial.lisp and run gradient
; descent by folding an update step over a sequence of iterations.
; Each step computes the gradient via (grad poly-loss) and subtracts
; a scaled version from the current parameters.
;
; The fold carries the parameter array as its accumulator:
;   (fold update-step initial-params (iota num-steps))
;
; The gradient function is the SAME compiled function at every step —
; only the input (current params) changes.  This is why source-to-source
; AD works: grad is resolved once at compile time, then called
; repeatedly at runtime.
;
; Data: x = [0, 1, 2, 3], y = [1, 2, 5, 10]
;
; Run:
;   remorac --syntax lisp --target interp examples/ad_optimize.lisp

; Loss function: sum of squared residuals for a degree-2 polynomial
(define/pi ()
  (poly-loss [coeffs (Array Float 3)] Float)
  (let* ((c0 (index-item coeffs 0))
         (c1 (index-item coeffs 1))
         (c2 (index-item coeffs 2))
         (r0 (- c0 1.0))
         (r1 (- (+ c0 (+ c1 c2)) 2.0))
         (r2 (- (+ c0 (+ (* c1 2.0) (* c2 4.0))) 5.0))
         (r3 (- (+ c0 (+ (* c1 3.0) (* c2 9.0))) 10.0)))
    (+ (* r0 r0) (+ (* r1 r1) (+ (* r2 r2) (* r3 r3))))))

; Gradient descent: 200 steps, learning rate 0.001
; Each step: params <- params - 0.001 * grad(loss)(params)
(fold (lambda (params step)
        (- params (* 0.001 ((grad poly-loss) params))))
      [0.0 0.0 0.0]
      (iota 200))
