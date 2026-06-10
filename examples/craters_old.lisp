;;;; ==========================================================================
;;;; LUNAR CRATER DETECTION IN REMORA
;;;; ==========================================================================

;;; --- Core Math & Activations ---

;; Element-wise ReLU
(define (relu x)
  (max 0.0 x))

;; Sigmoid activation for binary classification
(define (sigmoid x)
  (/ 1.0 (+ 1.0 (exp (- 0.0 x)))))

;; Binary Cross-Entropy Loss for a single patch
(define (bce-loss prediction target)
  (- 0.0 (+ (* target (log prediction))
            (* (- 1.0 target) (log (- 1.0 prediction))))))


;;; --- Network Layers ---

;; 2D Convolution: Applies a single KxK kernel to an MxN patch
;; Remora's rank lifting automatically handles multiple channels/kernels 
;; if shapes are structured as frames.
(define (conv2d patch kernel bias)
  (+ bias (reduce + 0.0 (* patch kernel))))

;; Fully Connected (Linear) Layer
;; Takes flattened features and weights, computes dot product and adds bias
(define (linear features weights bias)
  (+ bias (reduce + 0.0 (* features weights))))


;;; --- Forward Pass & Optimization ---

;; Complete forward pass for an image patch
;; params is an array containing: (kernel, conv-bias, fc-weights, fc-bias)
(define (forward params patch)
  (let* ((kernel    (idx params 0))
         (conv-bias (idx params 1))
         (fc-w      (idx params 2))
         (fc-b      (idx params 3))
         ;; 1. Conv layer (lifts naturally over spatial dimensions)
         (c-out     (relu (conv2d patch kernel conv-bias)))
         ;; 2. For simplicity, flatten or reduce spatial dimensions before FC
         (flat-feat (reshape (shape c-out) c-out)))
    (sigmoid (linear flat-feat fc-w fc-b))))

;; Objective function to pass to the automatic differentiation primitive
;; Returns the scalar mean loss across the entire batch
(define (loss-objective params batch-inputs batch-targets)
  (let* ((predictions (forward params batch-inputs))
         (losses       (bce-loss predictions batch-targets)))
    (reduce + 0.0 (/ losses (len batch-targets)))))

;; SGD Update Step using the built-in 'grad' AD primitive
(define (train-step params batch-inputs batch-targets learning-rate)
  (let* ((grads (grad loss-objective params batch-inputs batch-targets)))
    ;; Element-wise gradient descent update across the parameter structure
    (- params (* learning-rate grads))))


;;; --- Inference & Sliding Window Bounding Box Generator ---

;; Extract patches out of a large master image to map out coordinates
;; returns an array of shape [NumPatches, PatchHeight, PatchWidth]
(define (image->patches big-image patch-size stride)
  ;; Implemented via Remora's rerank and windowing capabilities
  ;; Expects a hyper-rectangular slice across the big-image canvas
  (window big-image (vector patch-size patch-size) (vector stride stride)))

;; Top-level application program
;; Inputs: big-image matrix, trained network parameters, and a detection threshold
;; Outputs: Coordinates of patches matching crater criteria
(define (detect-craters big-image params threshold)
  (let* ((patch-size 32)
         (stride     8)
         ;; Slice large image into an array of patches
         (patches    (image->patches big-image patch-size stride))
         ;; Generate patch coordinate maps corresponding to the window indices
         (coords     (generate-grid-coords (shape patches) stride))
         ;; LIFTED INFERENCE: forward evaluates across ALL patches automatically!
         (scores     (forward params patches))
         ;; Create a mask where scores exceed the detection threshold
         (mask       (> scores threshold)))
    ;; Filter and return coordinates paired with bounding boxes where mask is true
    (filter-coordinates coords mask patch-size)))

;;; --- Multi-Scale Image Pyramid Downsampler ---

;; Downsamples an image by a factor of 2 using a local average block reduction
(define (downsample-twice image)
  ;; Splits the image into 2x2 blocks and averages them
  (let* ((blocks (window image (vector 2 2) (vector 2 2))))
    (reduce + 0.0 (/ blocks 4.0))))


;;; --- Multi-Scale Crater Detection Application ---

;; Top-level function that recursively checks multiple image scales
(define (detect-craters-multiscale big-image params threshold scale-factor)
  (let* ((patch-size 32))
    (if (< (min (shape big-image)) patch-size)
        ;; Base case: Image has been downsampled so much it is smaller than our window
        (empty-array)
        
        ;; Recursive case: Detect at current scale, then downsample and detect again
        (let* ((current-detections (detect-craters big-image params threshold))
               ;; Project detections back to original scale by multiplying coordinates
               (scaled-detections  (* current-detections scale-factor))
               
               ;; Create the next tier of the pyramid
               (next-image         (downsample-twice big-image))
               (next-detections    (detect-craters-multiscale next-image 
                                                              params 
                                                              threshold 
                                                              (* scale-factor 2.0))))
          
          ;; Concatenate the bounding boxes found across all scales
          (concat scaled-detections next-detections)))))
