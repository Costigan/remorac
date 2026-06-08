; Rank polymorphism examples (Phase 2)
; Run: remorac --syntax lisp --target cpu examples/rank_polymorphism.lisp

; --- Scalar auto-lift ---
; Implicit map: (+ xs ys) auto-lifts to element-wise addition
(+ [1 2 3] [4 5 6])

; --- Scalar broadcasting ---
; Scalar replicated across array
(* 2 [1 2 3 4 5])

; --- Principal-frame broadcasting ---
; Vector broadcast to match matrix rows
(+ [10 20] [[1 2 3] [4 5 6]])

; --- Lambda auto-lift ---
; Inline lambda applied to array
((lambda (x) (* x 2)) [1 2 3])

; --- Vector-cell auto-lift ---
; Function defined with rank-1 cell annotation
(define (sum-vec [v 1]) (fold + 0 v))
(sum-vec [[1 2 3] [4 5 6]])
