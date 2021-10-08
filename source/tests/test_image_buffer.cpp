//
// Created by andreas on 08.10.21.
//
#include "gtest/gtest.h"
#include "image_buffer.h"

class SetupImageBuffer: public testing::Test
{
protected:
    int width{10};
    int height{15};
    int size{width*height};
    c_vector3 red_pixel{0.99, 0.05, 0.1};
    c_vector3 green_pixel{0.0, 0.99, 0.};
    c_vector3 blue_pixel{0.0, 0., 0.99};
};

TEST_F(SetupImageBuffer, test_image_buffer_width)
{
    auto image_buffer = ImageBuffer(width, height);
    EXPECT_EQ(image_buffer.width(), width);
    auto image_buffer_wrong = ImageBuffer(height, width);
    EXPECT_NE(image_buffer_wrong.width(), width);
}

TEST_F(SetupImageBuffer, test_image_buffer_height)
{
    auto image_buffer = ImageBuffer(width, height);
    EXPECT_EQ(image_buffer.height(), height);
    auto image_buffer_wrong = ImageBuffer(height, width);
    EXPECT_NE(image_buffer_wrong.height(), height);
}

TEST_F(SetupImageBuffer, test_image_buffer_size)
{
auto image_buffer = ImageBuffer(width, height);
auto buffer = image_buffer.buffer();
EXPECT_EQ(buffer->size(), size);
}


TEST_F(SetupImageBuffer, test_image_get_pixel)
{
    auto image_buffer = ImageBuffer(width, height);
    auto height_index =5;
    auto width_index =5;
    auto pixel  = image_buffer.get_rgb_pixel(width_index, height_index);
    for(int i =0 ; i <3; ++i)
    {
        EXPECT_EQ(pixel[i],0.0);
    }
    image_buffer.set_pixel_value(width_index,height_index, red_pixel);
    pixel  = image_buffer.get_rgb_pixel(width_index, height_index);
    for(int i =0 ; i <3; ++i)
    {
        float expected = (255 * std::max(0.f, std::min(1.f, red_pixel[i])));
        EXPECT_EQ(pixel[i],expected);
    }

}

TEST_F(SetupImageBuffer, test_image_set_pixel)
{
    auto image_buffer = ImageBuffer(width, height);
    auto height_index =3;
    auto width_index =4;
    image_buffer.set_pixel_value(width_index,height_index, blue_pixel);
    auto pixel  = image_buffer.get_rgb_pixel(width_index, height_index);
    for(int i =0 ; i <3; ++i)
    {
        float expected = (255 * std::max(0.f, std::min(1.f, blue_pixel[i])));
        EXPECT_EQ(pixel[i],expected);
    }
    image_buffer.set_pixel_value(width_index,height_index, green_pixel);
    pixel  = image_buffer.get_rgb_pixel(width_index, height_index);
    for(int i =0 ; i <3; ++i)
    {
        float expected = (255 * std::max(0.f, std::min(1.f, green_pixel[i])));
        EXPECT_EQ(pixel[i],expected);
    }

}
