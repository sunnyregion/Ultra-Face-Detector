#include "util.hpp"
#include <fstream>
#include "unistd.h"
#include <sys/time.h>

bool FileExists(const std::string &name)
{
    std::ifstream fhandle(name.c_str());
    return fhandle.good();
}

std::string GetCurPath()
{
    char buf[1024];
    getcwd(buf, 1024);
    return buf;
}

void sleep_ms(unsigned int secs)
{
    struct timeval tval;
    tval.tv_sec = secs / 1000;
    tval.tv_usec = (secs * 1000) % 1000000;
    select(0, NULL, NULL, NULL, &tval);
}

unsigned long get_cur_time(void)
{
    struct timeval tv;
    unsigned long ts;

    gettimeofday(&tv, NULL);

    ts = tv.tv_sec * 1000000 + tv.tv_usec;

    return ts;
}