#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "image_processing.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define RGBA_CHANNEL_COUNT 4
#define MAX_FILEPATH_SIZE 127

typedef struct img {
	int width;
	int height;
	int original_channel_count;
	int channel_count;
	uint8_t* data;
} img_t;

static img_t load_image(const char* filepath, int desired_channels) {
	img_t result;
	result.data = stbi_load(
		filepath,
		&result.width,
		&result.height,
		&result.original_channel_count,
		desired_channels
	);
	result.channel_count = desired_channels;
	return result;
}

#define MAX_FILEPATH_SIZE 127

void process_image(rgba_t* dst, rgba_t* src, uint32_t width, uint32_t height) {
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {

            uint32_t index = y * width + x;

			uint8_t rojo = src[index].r;
			uint8_t verde = src[index].g;
			uint8_t azul = src[index].b;

			uint8_t luminosidad = (uint8_t)(0.2126 * rojo + 0.7152 * verde + 0.0722 * azul);

			dst[index].r = luminosidad;
			dst[index].g = luminosidad;
			dst[index].b = luminosidad;
			dst[index].a = 255;         
        }
    }
}

int main(){
    char* filepath_src = "assets/procesador.png";
    char* filepath_dst = "assets/procesador_out.png";
	img_t src = load_image(filepath_src, RGBA_CHANNEL_COUNT);
	uint32_t width = src.width;
	uint32_t height = src.height;
	uint64_t out_size = width * height * sizeof(rgba_t);
	rgba_t* dst = malloc(out_size);
    process_image(
        dst,
        (rgba_t*) src.data,
        width, height
    );
	stbi_write_png(
		filepath_dst,
		width, height,
		RGBA_CHANNEL_COUNT,
		dst,
		width * sizeof(rgba_t)
	);
    free(dst);
	stbi_image_free(src.data);
}