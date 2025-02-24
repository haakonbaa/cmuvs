all:
	mkdir -p build && cd build && cmake .. && make

clean:
	rm -rf build

test:
	./build/tests/test_runner

.PHONY:
	all clean test
