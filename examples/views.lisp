; View and primitive examples (Phase 4 + Dense Core)
; Run: remorac --syntax lisp --target cpu examples/views.lisp

; --- Length ---
(length [10 20 30 40 50])
(length [[1 2] [3 4] [5 6]])

; --- Rotate ---
(rotate [1 2 3 4 5] 2)
(rotate [1 2 3 4 5] 0)

; --- Subarray ---
(subarray [[1 2 3] [4 5 6] [7 8 9]] [0 1] [2 2])

; --- Select ---
(select #t 42 99)
(select #f 42 99)

; --- Index-item ---
(index-item [10 20 30] 1)

; --- Indices-of ---
(indices-of [100 200 300])
(indices-of [[1 2] [3 4]])

; --- Reranking (Phase 5) ---
; ~(0 0) + is identity reranking: scalar cells
(map (~(0 0) +) [1 2 3] [4 5 6])
