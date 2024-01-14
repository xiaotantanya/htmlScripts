from bs4 import BeautifulSoup

# 假设原始HTML代码
html_code = '<html><head><title>Page Title</title><meta charset="UTF-8"></head><body><p>Content</p></body></html>'

# 使用BeautifulSoup解析HTML
soup = BeautifulSoup(html_code, 'html.parser')

# 创建要插入的新元素
new_element = soup.new_tag('link')
new_element['rel'] = 'stylesheet'
new_element['href'] = 'styles.css'

# 找到<header>元素
header_element = soup.head

# 在<header>内部的特定位置插入新元素
position_to_insert = 1  # 例如，在第二个位置插入
header_element.insert(position_to_insert, new_element)

# 打印修改后的HTML
print(soup.prettify())
