#ifndef UTIL_H_
#define UTIL_H_

#include <string>


struct face_landmark
{
    float x[5];
    float y[5];
};

struct face_box
{
    float x0;
    float y0;
    float x1;
    float y1;

    /* confidence score */
    float score;

    /*regression scale */
    float regress[4];

    /* padding stuff*/
    float px0;
    float py0;
    float px1;
    float py1;

    face_landmark landmark;
};

bool FileExists(const std::string &name);
std::string GetCurPath();
void sleep_ms(unsigned int secs);
/* get current time: in us */
unsigned long get_cur_time(void);

#endif // UTIL_H_