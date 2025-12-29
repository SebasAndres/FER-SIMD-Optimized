section .text

extern malloc

global euclideanDistanceASM
global centerVectorASM
global calculateMeanVectorASM
global dotProductASM

euclideanDistanceASM:
    ; float euclideanDistanceASM(
    ;   const float* vec1, 
    ;   const float* vec2, 
    ;   size_t length
    ; )
    ;
    ; Argumentos:
    ;   rdi = vec1 (const float*)
    ;   rsi = vec2 (const float*)
    ;   rdx = length (size_t)
    ; Output:
    ;   xmm0 = resultado (float)

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
    shr r14, 2   ; divido por 4 (shift right 2 bits)
    xor r13, r13  

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

        ; elevo al cuadrado cada float de xmm1
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


centerVectorASM:
    ; float* centerVectorASM(
    ;     float* vector, 
    ;     float* mean_vector,
    ;     size_t dim
    ; );
    ; 
    ; Argumentos:
    ;     rdi = vector, 
    ;     rsi = mean_vector
    ;     rdx = dim

    ; prólogo
    push rbp
    mov rbp, rsp

    ; guardo registros no volátiles
    ; RBX, RBP, R12, R13, R14 y R15
    push r12
    push r13
    push r14
    push r15      
    push rbx 
    sub rsp, 8

    ; guardo parámetros en regs. no volátiles
    mov r12, rdi ; vector 
    mov r13, rsi ; mean
    mov r14, rdx ; dim

    ; malloc(dim * sizeof(float))
    mov rdi, r14
    shl rdi, 2
    call malloc
    mov r15, rax

    ; iteraciones del loop = dim / num_floats_en_xmm
    mov r10, r14
    shr r10, 2

    xor r8, r8
    xor r9, r9
    .loop:
        movdqu xmm0, [r12 + r9]        
        movdqu xmm1, [r13 + r9]        
        subps xmm0, xmm1
   
        movdqu [r15 + r9], xmm0

        add r9, 16
        inc r8
        cmp r8, r10
        jne .loop

    ; retorno 
    mov rax, r15

    ; recupero registros no volátiles 
    add rsp, 8
    pop rbx
    pop r15
    pop r14
    pop r13
    pop r12

    ; epílogo
    pop rbp
    ret    

dotProductASM:
    ret

calculateMeanVectorASM:
    ret