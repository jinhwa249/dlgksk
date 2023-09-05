# list_number = [52, 273, 32, 72, 100]

# try:
#     number_input = int(input*"정수 입력> ")
#     print(f"{number_input}번째 요소: {list_number[number_input]}")
#     예외.발생해주세요()
# except ValueError as exception:
#     if exception is ValueError:
#         print("정수를 입력해 주세요!")
#     elif Exception is IndexError:
#         print("리스트의 인덱스를 벗어났어요!")
#     else:
#         print("미리 파악하지 못한 예외가 발생했습니다.")
#         print(type(exception),exception)

# number = input("정수 입력> ")
# number = int(number)

# if number > 0:
#     raise NotImplementedError
# else:
#     raise NotImplementedError
# import math as m

# print(m.sin(1))
# print(m.cos(1))
# print(m.tan(1))
# print(m.floor(2.5))
# print(m.ceil(2.5))

# import random
# print(" ramdom 모듈")

# print("random():",random.random())

# print("uniform(10,20):",random.uniform(10,20))


# print("randrange(10)",random.randrange(10))
# li = [1,2,3,4,5]

# print(f"choice.{li}:{random.choice(li)}")

# print(f"shufle.{li}:{random.shufle(li)}")

# print(f"sample{li}:{random.sample(li,k=2)}")

# import sys

# print(sys.argy)

# print("---")
      
# print(getwindowsvr)

# import os

# print("현제 운영체제", os.name)
# print("현제 폴더", os.listdir())
# print("현제 폴더 내부의 요소", os.listdir())
# os.mkdir("hello")
# os.rmdir("hello")

# with open("originak.txt", "w") as file:
#     file.write("hello")
    
# os.rename("original.txt", "new.txt")

# os.remove("new.txt")

# os.system("dir")

# import datetime

# print("현재 시간 출력하기")

# now = datetime.datetime.now()
# print(now.year, "년")
# print(now.month, "월")
# print(now.day, "일")
# print(now.hour, "시")
# print(now.minute, "분")
# print(now.second, "초")

# print()

# print("# 시간을 포맷에 맞춰 출력하기")
# output_a = now.strftime("%Y.%m.%d.%H.%M.%S")
# output_b = "{}년 {}월 {}일 {}시 {}분 {}초".format(now.year,\
#     now.month,\
#     now.day,\
#     now.hour,\
#     now.minute,\
#     now.second)
# output_c = now.strftime("%Y{}.%m{}.%d{}.%H{}.%M{}.%S{}").format(*"년월일시분초")
# print(output_a)
# print(output_b)
# print(output_c)
# print()

# import datetime
# now = datetime.datetime.now()

# print("# datetime.timedalta로 시간 더하기")
# after = now + datetime.timedelta(\
#     weeks=1,\
#     days=1,\
#     hours=1,\
#     minutes=1,\
#     seconds=1)
# print(after.strftime("%Y{}.%m{}.%d{}.%H{}.%M{}.%S{}").format(*"년월일시분초"))
# print()

# print("# now.replace()로 1년 더하기")
# output = now.replace(year=(now.year+1))
# print(output.strftime("%Y{}.%m{}.%d{}.%H{}.%M{}.%S{}").format(*"년월일시분초"))

# import time

# print("지금부처 5초동안 정지합니다.")

# time.sleep(5)

# print("프로그램을 종료합니다.")

from urllib import request

target = request.urlopen("https://google.com")

output = target.read()

print(output)
