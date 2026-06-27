; Linear interpolation example
; Translated from /e/projects/Remora/remora/examples/lerp.rkt
; Run: remorac --syntax lisp --target interp examples/lerp.lisp

; Find the number part way between two boundaries
(define (lerp lo hi α)
  (+ (* α hi)
     (* (- 1.0 α) lo)))

; Three fifths of the way from -1 to 1
(lerp -1.0 1.0 0.6)

; Try several "middle" points
(lerp -1.0 1.0 [0.0 0.333 0.667 1.0])

; Take the midpoint along multiple axes
(lerp [0.0 1.0] [7.0 4.0] 0.5)

; Find midpoints of three lines, formed by three low (x,y,z) points
; and one high (x,y,z) point.
; Reranking ~(1 1 0) means to treat (x,y,z) coordinates (rank-1
; structures) as the fundamental unit in lifting lerp.
((~(1 1 0) lerp)
 [[0.0 0.0 2.0] [0.0 1.0 1.0] [1.0 0.0 0.0]]
 [7.0 4.0 5.0]
 0.5)
