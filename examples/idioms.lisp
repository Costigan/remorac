; Array programming idioms
; Translated from /e/projects/Remora/remora/examples/idioms.rkt
;
; Includes: flat-apply, count-value, contains?, zero-to-n, int->bool.
; Excluded: character/string functions (no char support in remorac),
;   checkerboard (requires modulo), truth-table (requires expt/antibase),
;   histogram/range (require box/unbox with dynamic shapes),
;   num-digits (requires floor), move-blanks (chars), matrix-transpose
;   and position-matrix (rely on rank-polymorphic foldr patterns not
;   yet supported).
;
; Run: remorac --syntax lisp --target interp examples/idioms.lisp

; Ravel an array, apply a function to it, then reshape the new values
; to the shape of the original array.
(define (flat-apply op arr)
  (reshape (shape arr) (op (ravel arr))))

; Count the occurrences of a value in an array.
(define (count-value arr value)
  (fold-right + 0 (select (== value (ravel arr)) 1 0)))

; True if the array contains the given atom.
(define (contains? arr element)
  (fold-right || #f (== element (ravel arr))))

; Change zero values to n.
(define (zero-to-n v n)
  (select (== v 0) n v))

; Convert integer values to booleans (0 → #f, non-zero → #t).
(define (int->bool n)
  (select (== n 0) #f #t))

; --- Examples ---
(flat-apply (lambda (x) (* x 2)) [[1 2] [3 4]])

(count-value [1 2 3 1 2 1] 1)
(contains? [10 20 30 40] 30)
(contains? [10 20 30 40] 99)

(zero-to-n 0 42)
(zero-to-n 5 42)

(int->bool 0)
(int->bool 7)
