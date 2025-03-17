#include <pybind11/pybind11.h>
namespace py = pybind11;

// 迭代器类
class IntRangeIterator {
public:
    IntRangeIterator(int start, int end) : current(start), end(end) {}

    // 实现迭代器的递增和解引用操作
    int operator*() const { return current; }
    IntRangeIterator& operator++() {
        if (current < end) ++current;
        return *this;
    }
    bool operator!=(const IntRangeIterator& other) const {
        return current != other.current;
    }

private:
    int current;
    int end;
};

// 可迭代类
class IntRange {
public:
    IntRange(int start, int end) : start(start), end(end) {}

    // 返回迭代器的起点和终点
    IntRangeIterator begin() const { return IntRangeIterator(start, end); }
    IntRangeIterator end() const { return IntRangeIterator(end, end); }

private:
    int start;
    int end;
};