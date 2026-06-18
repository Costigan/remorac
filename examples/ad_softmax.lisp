; Automatic Differentiation Example 4: Softmax Cross-Entropy Loss
;
; The softmax cross-entropy loss is the standard training objective
; for classification.  Given logits z and one-hot targets t:
;   loss = log(sum(exp(z))) - sum(t * z)
;
; The gradient has the elegant form: softmax(z) - target.
; AD computes this automatically without manual derivation.
;
; Here: 3-class problem, target = class 0 → [1, 0, 0].
;
; Run:
;   remorac --syntax lisp --target interp examples/ad_softmax.lisp

(define/pi ()
  (cross-entropy [logits (Array Float 3)] Float)
  (let* ((z0 (index-item logits 0))
         (z1 (index-item logits 1))
         (z2 (index-item logits 2))
         (lse (log (+ (exp z0) (+ (exp z1) (exp z2))))))
    (- lse z0)))

; Gradient at uniform logits [0, 0, 0]
; Expected: softmax([0,0,0]) - [1,0,0] = [1/3-1, 1/3, 1/3] ≈ [-0.667, 0.333, 0.333]
((grad cross-entropy) [0.0 0.0 0.0])
