; Loops as syntactic sugar over letrec / tail recursion (Lisp syntax)
;
; Remora is a pure, rank-polymorphic array language, so its loops are
; *expressions that return a value* (the final accumulator) rather than
; imperative statements.  `letrec`, `while`, and `dotimes` all desugar to
; tail-recursive functions, so they run on the interpreter and the CPU
; backend, and a scalar loop used inside `map` also lowers to the GPU.
;
; Run:
;   remorac --syntax lisp --target interp examples/loops.lisp
;   remorac --syntax lisp --target cpu   examples/loops.lisp
;
; A condition-terminated `while` loop is just a tail-recursive `letrec`
; whose `if` tests the termination predicate (Newton's method, fixed
; points, ...).  `dotimes` is a bounded counter loop; `letrec` is the
; underlying local-recursion primitive that both desugar to.

; --- letrec: local (mutually) recursive helpers -------------------------
; n! using a tail-recursive accumulator helper.
(define/pi () (factorial [n Float] Float)
  (letrec ((go (lambda (k acc)
                 (if (< k 0.5) acc (go (- k 1.0) (* acc k))))))
    (go n 1.0)))
; (factorial 5.0)  =>  120.0

; letrec also supports mutual recursion: even?/odd? as 1.0/0.0.
(define/pi () (is-even [n Float] Float)
  (letrec ((even (lambda (k) (if (== k 0.0) 1.0 (odd (- k 1.0)))))
           (odd  (lambda (k) (if (== k 0.0) 0.0 (even (- k 1.0))))))
    (even n)))
; (is-even 8.0)  =>  1.0

; --- while: condition-terminated loop with threaded state ---------------
; Sum 1..n by counting k down while k > 0.  Bindings are (var init update)
; and update simultaneously each iteration; the loop returns the body.
(define/pi () (triangle [n Float] Float)
  (while (< 0.0 k)
    ((k   n   (- k 1.0))
     (acc 0.0 (+ acc k)))
    acc))
; (triangle 5.0)  =>  15.0

; Newton's method for sqrt(2): a `while` that stops on convergence.
(define/pi () (sqrt2 [x0 Float] Float)
  (while (< 0.0000001 (* (- (* x x) 2.0) (- (* x x) 2.0)))
    ((x x0 (- x (/ (- (* x x) 2.0) (* 2.0 x)))))
    x))
; (sqrt2 1.0)  =>  ~1.41421

; --- dotimes: bounded counter loop --------------------------------------
; Sum 0 + 1 + ... + (n-1).  The index runs 0..n-1; (acc init) is threaded.
(define/pi () (gauss [n Int] Int)
  (dotimes (i n) (acc 0) (+ acc i)))
; (gauss 5)  =>  10

; --- Loops compose with rank polymorphism (and lower to the GPU) --------
; Apply the scalar `triangle` loop to every element of an array; on a GPU
; target this lowers to a per-thread tail-recursion state machine.
(define/pi () (triangles [xs (Array Float 4)] (Array Float 4))
  (map (lambda (x) (triangle x)) xs))
; (triangles [1.0 2.0 3.0 4.0])  =>  [1.0 3.0 6.0 10.0]

; Trailing expression — this is what running the file prints.
(triangle 5.0)
