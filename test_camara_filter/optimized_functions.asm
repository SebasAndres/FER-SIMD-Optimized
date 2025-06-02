
section .data
    OFFSET_FRAME_DATA equ 0
    OFFSET_FRAME_ROWS equ 8
    OFFSET_FRAME_COLS equ 12

section .text

; void apply_filter_asm(cv::Mat& image);
global apply_filter_asm
apply_filter_asm:    

    ; Arguments:
    ; rdi: pointer to cv::Mat

    push rbp
    mov rbp, rsp

    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8
    
    ; Get the pointer to the image data
    mov rax, [rdi + OFFSET_FRAME_DATA] ; pointer to data
    mov rbx, rax ; rdi = pointer to image data

    ; Get the number of rows and columns
    mov esi, dword [rdi + OFFSET_FRAME_COLS] ; number of columns
    mov edx, dword [rdi + OFFSET_FRAME_ROWS] ; number of rows

    xor r8d, r8d; row
    .row_loop:
        xor r9d, r9d ; column

        .col_loop:
        
            xor rax, rax
            mov eax, r8d
            imul eax, esi     
            add eax, r9d      
            imul eax, eax, 3  

            mov r10b, byte [rbx + rax] 

            ; mov [rbx + rax], byte 0 

            inc r9d
            cmp r9d, esi
            jl .col_loop

        inc r8d
        cmp r8d, edx
        jl .row_loop

    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx

    pop rbp
    ret