; Automatic Differentiation Example 2: Circle Fitting
;
; Fit a circle (center x, center y, radius) to 2D points by minimizing
; the sum of squared "distance-squared minus radius-squared" residuals.
;
; For each point, residual = (px-cx)^2 + (py-cy)^2 - r^2.
; This avoids sqrt while producing the same optimum.
;
; Data points (on a circle of radius 5 centered at (3, 4)):
;   (8,4), (3,9), (-2,4), (3,-1)
;
; Run:
;   remorac --syntax lisp --target interp examples/ad_circle.lisp

(define/pi ()
  (circle-loss [params (Array Float 3)] Float)
  (let* ((cx (index-item params 0))
         (cy (index-item params 1))
         (r  (index-item params 2))
         (r2 (* r r))
         (e0 (- (+ (* (- 8.0 cx) (- 8.0 cx)) (* (- 4.0 cy) (- 4.0 cy))) r2))
         (e1 (- (+ (* (- 3.0 cx) (- 3.0 cx)) (* (- 9.0 cy) (- 9.0 cy))) r2))
         (e2 (- (+ (* (+ 2.0 cx) (+ 2.0 cx)) (* (- 4.0 cy) (- 4.0 cy))) r2))
         (e3 (- (+ (* (- 3.0 cx) (- 3.0 cx)) (* (+ 1.0 cy) (+ 1.0 cy))) r2)))
    (+ (* e0 e0) (+ (* e1 e1) (+ (* e2 e2) (* e3 e3))))))

; Gradient at the true center [3, 4, 5] — should be near zero
((grad circle-loss) [3.0 4.0 5.0])
