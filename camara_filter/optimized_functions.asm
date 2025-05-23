section .text 

extern malloc 

global apply_filter_asm

apply_filter_asm:    
    ; char* data [rdi]
    ; uint32_t width [esi]
    ; uint32_t height [edx]

    push rbp
    mov rbp, rsp

    xor r8, r8 ; row
    .row_loop:
        ; Calculate the row offset
        mov rax, r8
        mul esi 
        mov r10, rax ; r10 = row offset

        xor r9, r9 ; column
        .col_loop:
            ; Calculate how many bytes to process (up to 16 at a time)
            mov rcx, rsi
            sub rcx, r9
            cmp rcx, 16
            jl .process_tail

            ; SIMD: process 16 bytes        
            lea r11, [r10 + r9]
            add r11, rdi
            movdqu xmm0, [r11]         ; Use movdqu for unaligned memory access
            pcmpeqb xmm1, xmm1         ; xmm1 = 0xFF..FF
            pxor xmm0, xmm1            ; invert all bytes
            movdqu [r11], xmm0         ; Use movdqu for unaligned memory access

            add r9, 16
            cmp r9, rsi
            jl .col_loop
            jmp .next_row

        .process_tail:
            ; Process remaining bytes one by one
            cmp rcx, 0
            jle .next_row
            lea r11, [r10 + r9]
            add r11, rdi
            movzx r12, byte [r11]
            xor r12b, 0xFF
            mov [r11], r12b
            inc r9
            cmp r9, rsi
            jl .process_tail

        .next_row:
        inc r8
        cmp r8, rdx
        jl .row_loop

    pop rbp
    ret