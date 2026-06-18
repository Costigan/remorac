; Automatic Differentiation Example 1: Polynomial Curve Fitting
;
; Fit a degree-2 polynomial f(x) = c0 + c1*x + c2*x^2 to data points
; by minimizing the sum of squared residuals.
;
; The gradient of the loss with respect to the coefficients tells us
; how to adjust each coefficient to reduce the fitting error.
;
; Data: x = [0, 1, 2, 3], y = [1, 2, 5, 10]
;
; Run:
;   remorac --syntax lisp --target interp examples/ad_polynomial.lisp
;
; --- How the AD works ---
;
; The function poly-loss below is plain forward arithmetic — no AD
; annotations, no manual derivatives.  All the differentiation happens
; in the single expression (grad poly-loss) on the last line.
;
; (grad poly-loss) transforms the function into its gradient.  The
; interpreter:
;   1. Traces poly-loss at the given input, executing it forward while
;      recording every primitive operation (+, *, -, index-item) on an
;      evaluation tape.
;   2. Reverses the tape, propagating adjoints backward from the output
;      (d(loss)/d(loss) = 1) to the input (d(loss)/d(coeffs)) using the
;      chain rule for each recorded operation.  For z = x * y, the
;      adjoint of x accumulates adjoint(z) * y; for z = x + y, both
;      accumulate adjoint(z).
;
; ((grad poly-loss) [0 0 0]) then calls the resulting gradient function
; at [0, 0, 0], returning [-36, -84, -224] — the partial derivatives
; d(loss)/d(c0), d(loss)/d(c1), d(loss)/d(c2).
;
; The implementation lives in remora/ad.py (tape tracing and reverse-mode
; backward pass) and remora/ad_source.py (source-to-source gradient
; generation for the compiled CPU/GPU path).

; --- Forward loss function (no AD here, just arithmetic) ---

(define/pi ()
  (poly-loss [coeffs (Array Float 3)] Float)
  (:: c0 (index-item coeffs 0)
  (:: c1 (index-item coeffs 1)
  (:: c2 (index-item coeffs 2)
    (:: r0 (- c0 1.0)
    (:: r1 (- (+ c0 (+ c1 c2)) 2.0)
    (:: r2 (- (+ c0 (+ (* c1 2.0) (* c2 4.0))) 5.0)
    (:: r3 (- (+ c0 (+ (* c1 3.0) (* c2 9.0))) 10.0)
      (+ (* r0 r0) (+ (* r1 r1) (+ (* r2 r2) (* r3 r3))))))))))))

; --- AD happens here: grad traces, reverses, returns the gradient ---

((grad poly-loss) [0.0 0.0 0.0])
