# Compiler and flags
CC      := clang
CFLAGS  := -O3
LDFLAGS := -lm

# Target and sources
TARGET  := main
SRCS    := main.c

# Default target
all: $(TARGET)

# Build rule
$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) $(LDFLAGS) -o $(TARGET)

# Clean up build artifacts
clean:
	rm -f $(TARGET)

.PHONY: all clean

