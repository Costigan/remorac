; Automatic Differentiation Example 3: Spring Forces from Potential Energy
;
; A chain of 4 particles connected by springs.  The gradient of the
; total energy with respect to positions gives the force on each
; particle (with sign flip): F = -dV/dx.
;
; Setup: k=2, rest length=1.  Positions = [x0, x1, x2, x3].
;
; Run:
;   remorac --syntax lisp --target interp examples/ad_spring.lisp

(define/pi ()
  (chain-energy [pos (Array Float 4)] Float)
  (:: x0 (index-item pos 0)
  (:: x1 (index-item pos 1)
  (:: x2 (index-item pos 2)
  (:: x3 (index-item pos 3)
    (:: e01 (* (- (- x1 x0) 1.0) (- (- x1 x0) 1.0))
    (:: e12 (* (- (- x2 x1) 1.0) (- (- x2 x1) 1.0))
    (:: e23 (* (- (- x3 x2) 1.0) (- (- x3 x2) 1.0))
      (+ e01 (+ e12 e23))))))))))

; Gradient at compressed state [0, 0.5, 1, 1.5] — restoring forces
((grad chain-energy) [0.0 0.5 1.0 1.5])
