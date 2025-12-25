section .data
    ; Constante para testing
    test_value: dd 42.0

section .text
global euclideanDistanceASM

; float euclideanDistanceASM(const float* vec1, const float* vec2, size_t length)
; Argumentos (System V AMD64 ABI):
;   rdi = vec1 (const float*)
;   rsi = vec2 (const float*)
;   rdx = length (size_t)
; Retorno:
;   xmm0 = resultado (float)

euclideanDistanceASM:
    ; Por ahora retorna 42.0 para verificar que linkea bien
    movss xmm0, [rel test_value]
    ret
