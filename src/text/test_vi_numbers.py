"""Test cho vi_numbers. Chạy: python -m unittest src.text.test_vi_numbers -v
Các câu lấy từ transcript train và cách gold ghi argument tương ứng."""
import unittest

from src.text.vi_numbers import asr_target, itn, llm_input, read_number, verbalize


class TestReadNumber(unittest.TestCase):
    def test_cases(self):
        cases = {
            0: "không", 5: "năm", 10: "mười", 15: "mười lăm", 21: "hai mươi mốt", 24: "hai mươi tư",
            25: "hai mươi lăm", 50: "năm mươi", 105: "một trăm linh năm", 123: "một trăm hai mươi ba",
            500: "năm trăm", 1050: "một nghìn không trăm năm mươi", 2026: "hai nghìn không trăm hai mươi sáu",
            12000: "mười hai nghìn", 2800000: "hai triệu tám trăm nghìn",
        }
        for n, words in cases.items():
            self.assertEqual(read_number(n), words, n)


class TestVerbalize(unittest.TestCase):
    def test_cases(self):
        cases = {
            "Nạp 50 nghìn nhé.": "Nạp năm mươi nghìn nhé.",
            "dưới 123 triệu thì 9 giờ sáng mai": "dưới một trăm hai mươi ba triệu thì chín giờ sáng mai",
            "lúc 8 giờ kém 15 tối mai": "lúc tám giờ kém mười lăm tối mai",
            "số 0983 214 6755 là": "số không chín tám ba hai một bốn sáu bảy năm năm là",
            "Vé anh số 412750, dò": "Vé anh số bốn một hai bảy năm không, dò",
            "gọi 09889990002 nhé": "gọi không chín tám tám chín chín chín không không không hai nhé",
            # mã chữ + số giữ nguyên
            "mở VTV6 lên, đổ RON95, chuyến RV219, Góc Phố 24h, lớp 8A3, Omega-3, giấy A4":
                "mở VTV6 lên, đổ RON95, chuyến RV219, Góc Phố 24h, lớp 8A3, Omega-3, giấy A4",
        }
        for src, out in cases.items():
            self.assertEqual(verbalize(src), out, src)

    def test_asr_target(self):
        self.assertEqual(asr_target("Ờ… tắt đèn tầng 2, check-in giúp nhé!"), "ờ tắt đèn tầng hai check in giúp nhé")


class TestITN(unittest.TestCase):
    def test_amounts(self):
        cases = {
            "chuyển hai triệu tám cho chủ nhà": "chuyển 2800000 cho chủ nhà",
            "nạp hai trăm nghìn": "nạp 200000",
            "nạp năm mươi nghìn điện thoại": "nạp 50000 điện thoại",
            "gửi bốn chục triệu": "gửi 40000000",
            "hơn mười hai triệu rưỡi thì": "hơn 12500000 thì",
            "đổi năm trăm đô": "đổi 500 đô",
            "mục tiêu mười hai ngàn bước": "mục tiêu 12000 bước",
            "trừ năm chục số thôi": "trừ 50 số thôi",
            "hai nghìn rưỡi": "2500",
            "một trăm linh năm": "105",
            "dưới một trăm hai mươi ba triệu": "dưới 123000000",
        }
        for src, out in cases.items():
            self.assertEqual(itn(src), out, src)

    def test_time_units(self):
        cases = {
            "chín giờ kém mười lăm tối mai": "9 giờ kém 15 tối mai",
            "năm rưỡi chiều nay": "5 rưỡi chiều nay",
            "bật điều hoà hai mươi sáu độ": "bật điều hoà 26 độ",
            "mờ năm mươi phần trăm": "mờ 50 phần trăm",
            "tắt đèn tầng hai": "tắt đèn tầng 2",
            "đặt hai chỗ": "đặt 2 chỗ",
            "đi bộ ba mươi lăm phút": "đi bộ 35 phút",
            "trong năm ngày tới": "trong 5 ngày tới",
            "ba mét dây": "3 mét dây",
            "thứ tư mùng bảy": "thứ tư mùng 7",
            "muộn một tiếng": "muộn 1 tiếng",
            # cách nói gọn
            "giá xăng chín lăm hôm nay": "giá xăng 95 hôm nay",
            "điều hoà hai tư độ": "điều hoà 24 độ",
            "điều hoà hai sáu độ nhé": "điều hoà 26 độ nhé",
            "đi bộ ba lăm phút": "đi bộ 35 phút",
            "hai mốt giờ": "21 giờ",
        }
        for src, out in cases.items():
            self.assertEqual(itn(src), out, src)

    def test_digit_sequences(self):
        cases = {
            "số không chín năm không năm ba sáu không ba sáu bảy nha": "số 09505360367 nha",
            "số một tám năm sáu không không trúng thì": "số 185600 trúng thì",
        }
        for src, out in cases.items():
            self.assertEqual(itn(src), out, src)

    def test_keep_ambiguous_words(self):
        for s in ["năm nay mưa nhiều", "ngày mốt", "ba mẹ về chưa", "một chút thôi", "một nửa", "không mưa thì thôi",
                  "đi một mình", "vài chục người", "hàng trăm", "hôm hai chín ấy", "hai ba ngày nữa",
                  "thứ hai tư vấn"]:
            self.assertEqual(itn(s), s, s)

    def test_roundtrip_digits(self):
        # số 1 chữ số đứng một mình được giữ nguyên chữ (đa nghĩa), nên chỉ kiểm tra từ 10
        for n in [10, 15, 21, 24, 26, 35, 50, 99, 100, 105, 120, 250, 500, 999, 1050, 2026, 9380, 12000,
                  200000, 2800000, 40000000, 123000000]:
            self.assertEqual(itn(read_number(n)), str(n), n)

    def test_llm_input(self):
        self.assertEqual(llm_input("Nạp dùm em Linh năm mươi ngàn, cho số không chín không một, hai ba bốn, chín chín chín sáu."),
                         "nạp dùm em linh 50000 cho số 09012349996")


if __name__ == "__main__":
    unittest.main()
