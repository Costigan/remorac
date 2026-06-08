;; Remora Integration Example — exercises Phases 1-5
;; Run: remorac --syntax lisp --target cpu examples/integration.lisp

;; Phase 3: Scan/reduce/fold
(iscan + 0 [1 2 3 4 5])

;; Phase 4: Primitives  
(append [1 2 3] [4 5 6])
(rotate [1 2 3 4 5] 2)
(subarray [[1 2 3] [4 5 6] [7 8 9]] [1 0] [2 2])
(indices-of [100 200 300])
(with-shape 7 [2 3])
(length [10 20 30 40 50])
(select #t 42 99)

;; Phase 2: Rank polymorphism (scalar + lambda auto-lift)
(+ [1 2 3] [4 5 6])
(* 2 [1 2 3 4])
((lambda (x) (* x 2)) [1 2 3])

;; Phase 5: Reranking
(map (~(0 0) +) [1 2 3] [4 5 6])
