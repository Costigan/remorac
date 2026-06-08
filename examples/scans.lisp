; Scan/reduce/fold examples (Phase 3)
; Run: remorac --syntax lisp --target cpu examples/scans.lisp

; --- Reduce (like fold but explicitly parallel) ---
(reduce + 0 (iota 10))

; --- Inclusive scan ---
(iscan + 0 [2 10 5])

; --- Exclusive scan ---
(escan + 0 [2 10 5])

; --- Fold-right ---
(fold-right + 0 [1 2 3 4])

; --- Trace (serial scan) ---
(trace + 0 [2 10 5])

; --- Trace-right (reverse scan) ---
(trace-right + 0 [2 10 5])

; --- Reduce variants ---
(reduce/zero + 0 [1 2 3])
(reduce/1 + 0 [1 2 3])

; --- Scan variants ---
(iscan/zero + 0 [2 10 5])
(iscan/1 + 0 [2 10 5])
