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
    ; prólogo
    push rbp
    mov rbp, rsp

    ; guardo registros no volátiles
    push r13
    push r14
    push r15      
    sub rsp, 8   
    
    ; guardo argumentos iniciales
    mov r15, rdx ; tamaño vectores

    ; cantidad de iteraciones del loop auxiliar (4 floats a la vez)
    mov r14, rdx 
    shr r14, 2   ; dividir por 4 (shift right 2 bits)
    xor r13, r13 ; iterador 

    ; acumulador de distancias
    pxor xmm15, xmm15

    .loop:
        ; calculo offset
        ; offset = indice * 16
        mov rax, r13
        shl rax, 4   
        
        ; leo parámetros
        movdqu xmm1, [rdi + rax] 
        movdqu xmm2, [rsi + rax]      

        ; resta vertical (z_i = x_i - y_i)  
        subps xmm1, xmm2

        ; aplicar cuadrados (z_i = z_i**2)
        ; elevar al cuadrado cada float de xmm1
        mulps xmm1, xmm1

        ; sumas horizontales (z = z_1 + z_2 + z_3 + z_4)
        haddps xmm1, xmm1
        haddps xmm1, xmm1

        ; vacio los bits superiores de xmm1
        movss xmm1, xmm1         

        ; sumamos al acumulador
        addps xmm15, xmm1

        ; siguiente iteración
        inc r13
        cmp r13, r14
        jne .loop 

    ; raíz cuadrada del float 
    sqrtss xmm15, xmm15

    ; retorno
    movss xmm0, xmm15

    ; epílogo
    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop rbp
    ret    