from PySide6.QtGui import QFont
import silicon

def adjusted(p):
    return int(p/silicon.SA_SCALE_FACTOR)

font_name = ['微软雅黑']

font_L1 = QFont(font_name, adjusted(10), QFont.Normal)
font_L1_bold = QFont(font_name,adjusted(10), QFont.Bold)

font_L2 = QFont(font_name,adjusted(14), QFont.Normal)
font_L2_bold = QFont(font_name,adjusted(14), QFont.Bold)

font_L3 = QFont(font_name,adjusted(18), QFont.Normal)
font_L3_bold = QFont(font_name,adjusted(18), QFont.Bold)

font_L4 = QFont(font_name,adjusted(24), QFont.Normal)
font_L4_bold = QFont(font_name,adjusted(24), QFont.Bold)

print(font_L1, font_L2)

def fontraw(name, size, tag = QFont.Normal):
    return QFont(name ,size, tag)
