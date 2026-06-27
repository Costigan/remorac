; Digital signal processing operations
; Translated from /e/projects/Remora/remora/examples/dsp.rkt
;
; Includes: simple-low-pass, simple-high-pass, fir-filter, dot.
; Excluded: IIR low-pass/high-pass (rely on head/tail array operations
;   not available in remorac), DFT/Goertzel (require cos, pi, complex
;   numbers, position-matrix), convolution with provided kernel matrices
;   (the filter coefficient constants are defined but the FIR filter
;   function is the portable core).
;
; Run: remorac --syntax lisp --target interp examples/dsp.lisp

; Generalized dot product: element-wise multiply then sum.
(define (dot a b)
  (fold-right + 0 (* a b)))

; Simple low-pass filter: y(n) = x(n) + x(n-1) with x(-1) = seed.
; The seed parameter sets the "virtual" sample before the first sample.
(define (simple-low-pass seed data)
  (let* ((n (length data))
         ; Shift data left by 1, drop the wrapped-around element, pad with seed.
         (shifted (append (take (- n 1) (rotate data 1)) [seed])))
    (+ data shifted)))

; Simple high-pass filter: y(n) = x(n) - x(n-1) with x(-1) = seed.
(define (simple-high-pass seed data)
  (let* ((n (length data))
         (shifted (append (take (- n 1) (rotate data 1)) [seed])))
    (- data shifted)))

; General FIR filter.
; y(n) = sum_{k=0}^{M-1} h(k) * x(n-k)
; Where h = coeffs (length M) and x = data.
(define (fir-filter coeffs data)
  (let* ((n (length coeffs))
         (shifts (iota n))
         (rotated (map (lambda (i) (rotate data i)) shifts)))
    (fold-right + 0 (* coeffs rotated))))

; --- FIR coefficient examples from http://t-filter.engineerjs.com/ ---
(define low-pass-kernel
  [-0.02010411882885732
   -0.05842798004352509
   -0.061178403647821976
   -0.010939393385338943
   0.05125096443534972
   0.033220867678947885
   -0.05655276971833928
   -0.08565500737264514
   0.0633795996605449
   0.310854403656636
   0.4344309124179415
   0.310854403656636
   0.0633795996605449
   -0.08565500737264514
   -0.05655276971833928
   0.033220867678947885
   0.05125096443534972
   -0.010939393385338943
   -0.061178403647821976
   -0.05842798004352509
   -0.02010411882885734])

; --- Examples ---
(dot [1.0 2.0 3.0] [4.0 5.0 6.0])

(simple-low-pass 0 [1.0 2.0 3.0 4.0 5.0])
(simple-high-pass 0 [1.0 2.0 3.0 4.0 5.0])

; FIR filter with a simple [0.2 0.5 0.2] kernel (moving average-ish)
(fir-filter [0.2 0.5 0.2] [1.0 2.0 3.0 4.0 5.0])
