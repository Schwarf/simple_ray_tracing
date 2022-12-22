//
// Created by andreas on 08.10.21.
//

#include "gtest/gtest.h"
#include "miscellaneous/templates/fixed_sized_array.h"
#include <regex>

class SetupCVector: public testing::Test
{
protected:
	double x1{1.2111111999};
	double x2{-1.731131};
	double x3{2.612123121};
	double y1{-21.99773131};
	double y2{-2.151890121};
	double y3{3.81123191977};
	double factor = 4.518291890;
};

TEST_F(SetupCVector, test_index_operator)
{
	FixedSizedArray<2, double> double_vector{x1, x2};
	EXPECT_DOUBLE_EQ(x1, double_vector[0]);
	EXPECT_DOUBLE_EQ(x2, double_vector[1]);

	try {
		double_vector[2];
		FAIL() << "Expected std::out_of_range";
	}
	catch (std::out_of_range const &err) {
		EXPECT_TRUE(std::regex_match(err.what(), std::regex("In class FixedSizedArray 'index (.*)")));
	}
}

TEST_F(SetupCVector, test_norm)
{
	FixedSizedArray<4, double> double_vector{x1, x2, y1, y2};
	auto norm = double_vector.norm();
	auto expected_value = std::sqrt(x1 * x1 + x2 * x2 + y1 * y1 + y2 * y2);
	EXPECT_DOUBLE_EQ(norm, expected_value);
}

TEST_F(SetupCVector, test_normalize)
{
	FixedSizedArray<4, double> double_vector{x1, x2, y1, y2};
	auto normalized = double_vector.normalize().norm();
	auto normalized2 = double_vector.normalize() * double_vector.normalize();
	auto expected_value = 1.0;
	EXPECT_DOUBLE_EQ(normalized, expected_value);
	EXPECT_DOUBLE_EQ(normalized2, expected_value);
}


TEST_F(SetupCVector, test_multiplication_operator_for_factor)
{
	FixedSizedArray<3, double> double_vector{x1, x2, y1};
	auto result = factor * double_vector;
	auto reference_vector = FixedSizedArray<3, double>{x1 * factor, x2 * factor, y1 * factor};
	for (int i = 0; i < 3; ++i) {
		EXPECT_DOUBLE_EQ(reference_vector[i], result[i]);
	}
	FixedSizedArray<3, double> double_vector2{x2, x3, y3};
	auto result2 = factor * double_vector2;
	auto reference_vector2 = FixedSizedArray<3, double>{x2 * factor, x3 * factor, y3 * factor};
	for (int i = 0; i < 3; ++i) {
		EXPECT_DOUBLE_EQ(reference_vector2[i], result2[i]);
	}
	result = factor * double_vector;
}

TEST_F(SetupCVector, test_division_operator_for_factor)
{
	FixedSizedArray<3, double> double_vector{x1, x2, y1};
	auto result = double_vector / factor;
	auto reference_vector = FixedSizedArray<3, double>{x1 / factor, x2 / factor, y1 / factor};
	for (int i = 0; i < 3; ++i) {
		EXPECT_DOUBLE_EQ(reference_vector[i], result[i]);
	}
}

TEST_F(SetupCVector, test_multiplication_operator_dot_product)
{
	FixedSizedArray<2, double> double_vector1{x1, x2};
	FixedSizedArray<2, double> double_vector2{y2, y3};
	auto expected_result = x1 * y2 + x2 * y3;
	EXPECT_DOUBLE_EQ(double_vector1 * double_vector2, expected_result);
}

TEST_F(SetupCVector, test_addition_operator)
{
	FixedSizedArray<2, double> double_vector1{x1, x2};
	FixedSizedArray<2, double> double_vector2{y2, y3};
	auto result = double_vector1 + double_vector2;
	auto expected_result = FixedSizedArray<2, double>{x1 + y2, x2 + y3};
	for (int i = 0; i < 2; ++i) {
		EXPECT_DOUBLE_EQ(result[i], expected_result[i]);
	}

}

TEST_F(SetupCVector, test_subtraction_operator)
{
	FixedSizedArray<2, double> double_vector1{x1, x2};
	FixedSizedArray<2, double> double_vector2{y2, y3};
	auto result = double_vector1 - double_vector2;
	auto expected_result = FixedSizedArray<2, double>{x1 - y2, x2 - y3};
	for (int i = 0; i < 2; ++i) {
		EXPECT_DOUBLE_EQ(result[i], expected_result[i]);
	}

}

TEST_F(SetupCVector, test_sign_operator)
{
	FixedSizedArray<2, double> double_vector1{x1, x2};
	auto result = -double_vector1;
	auto expected_result = FixedSizedArray<2, double>{-x1, -x2};
	for (int i = 0; i < 2; ++i) {
		EXPECT_DOUBLE_EQ(result[i], expected_result[i]);
	}

}

TEST_F(SetupCVector, test_cross_product)
{
	FixedSizedArray<3, float> float_vector1{float(x1), float(x2), float(x3)};
	FixedSizedArray<3, float> float_vector2{float(y1), float(y2), float(y3)};
	auto expected_result = FixedSizedArray<3, float>{-0.97673979, -62.07660823, -40.68713283};
	auto result = cross_product(float_vector1, float_vector2);
	for (int i = 0; i < 3; ++i) {
		EXPECT_FLOAT_EQ(result[i], expected_result[i]);
	}

}
