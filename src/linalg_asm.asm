section .data
    mask_1 dd 1.0

section .text

extern malloc
extern calloc

global euclideanDistanceASM
global centerVectorASM
global calculateMeanVectorASM
global dotProductASM

SIZE_POINTER EQU 8

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

        ; sumamos al acumulador
        addss xmm15, xmm1

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
    ; float dotProductASM(
    ;     const float* vec1,
    ;     const float* vec2,
    ;     size_t length
    ; )
    ; 
    ; Argumentos:
    ;     rdi = vec1, 
    ;     rsi = vec2
    ;     rdx = length
    
    ;prólogo
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

    ; contador del loop
    xor rcx, rcx
    xor r9, r9
    shr rdx, 2 

    ; output
    pxor xmm0, xmm0

    .loop:
        movdqu xmm1, [rdi + r9]
        movdqu xmm2, [rsi + r9]

        mulps xmm1, xmm2
        addps xmm0, xmm1

        add r9, 16
        inc rcx
        cmp rcx, rdx
        jne .loop

    ; al final hago la suma total
    haddps xmm0, xmm0
    haddps xmm0, xmm0

    ; recupero registros no volátiles 
    add rsp, 8
    pop rbx
    pop r15
    pop r14
    pop r13
    pop r12

    ;epílogo
    pop rbp
    ret

calculateMeanVectorASM:
    ; float* calculateMeanVectorASM(
    ;     float** vectors,
    ;     size_t num_vectors,
    ;     size_t vector_dim
    ; )
    ; Argumentos:
    ;     rdi = vectors, 
    ;     rsi = num_vectors
    ;     rdx = vector_dim
 
    ;prólogo
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

    ; guardo los argumentos
    mov r12, rdi ; vectors
    mov r13, rsi ; num_vectors
    mov r14, rdx ; vector_dim

    ; pido memoria para la respuesta
    ; un vector de floats de dimension vector_dim
    mov rdi, rdx
    mov rsi, 4
    call calloc ; output en rax

    ; cantidad de iteraciones en loop vectores
    mov rbx, r13 

    ; cantidad de iteraciones por vector
    shr r14, 2

    ; i
    xor r9, r9

    .loop_vectors:  
        ;vectors[i]
        mov r10, [r12 + r9 * SIZE_POINTER]        

        ;j
        xor r8, r8
        xor rcx, rcx
        .loop_inside_vector:

            ; vectors[i,j]
            movdqu xmm1, [r10 + rcx]     

            ; acumulado
            movdqu xmm0, [rax + rcx]
            
            ; nuevo acumulado
            addps xmm0, xmm1
            movdqu [rax + rcx], xmm0 
            
            ; actualizo iteradores del loop interno
            inc r8
            add rcx, 16
            cmp r8, r14
            jne .loop_inside_vector

        ; actualizo iteradores del loop externo
        inc r9 
        cmp r9, rbx 
        jne .loop_vectors

    ; armo un xmm con 4 floats que valen 1/num_vectors
    cvtsi2ss xmm4, r13 
    movss xmm5, [mask_1]
    divss xmm5, xmm4 
    shufps xmm5, xmm5, 0 

    ; divido el vector resultante por num_vectors
    xor r8, r8
    xor rcx, rcx
    .loop_inside_vector_div:
        movdqu xmm0, [rax + rcx]
        mulps xmm0, xmm5
        movdqu [rax + rcx], xmm0
        inc r8 
        add rcx, 16
        cmp r8, r14
        jne .loop_inside_vector_div
    
    ; recupero registros no volátiles 
    add rsp, 8
    pop rbx
    pop r15
    pop r14
    pop r13
    pop r12

    ;epílogo
    pop rbp
    ret
