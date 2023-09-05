# from urllib import request
# from bs4 import BeautifulSoup
   
# target = request.urlopen("http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp?stnId=108")
# soup = BeautifulSoup(target, "html.parser")

# for location in soup.select("location"):
#     print("도시:", location.select_one("city").string)
#     print("날씨:", location.select_one("wf").string)
#     print("최저기온:", location.select_one("tmn").string)
#     print("최고기온:", location.select_one("tmx").string)
#     print()
# from urllib import request
# from bs4 import BeautifulSoup
# from  flask import Flask

#     # return "<h1>Hello World!</h1>"
# app = Flask(__name__)

# @app.route("/")
# def hello():
#     target = request.urlopen("http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp?stnId=108")
#     soup = BeautifulSoup(target, "html.parser")
#     output = ""
#     for location in soup.select("location"):
#         output += "<h3>{}</h3>".format(location.select_one("city").string)
#         output += "날씨: {}<br/>".format(location.select_one("wf").string)
#         output += "최저/최고 기온: {}/{}"\
#             .format(\
#                 location.select_one("tmn").string,\
#                 location.select_one("tmx").string\
#             )
#         output += "<hr/>"
#     return output

# PI=3.141592

# def number_imput():
#     output= input("숫자 입력> ")
#     return float(output)

# def get_circumference(radius):
#     return 2 * PI * radius

# def get_circumference(radius):
#     return PI * radius * radius

# import teat_module as test

# radius = test.number_input()
# print(test.get_circumference(radius))
# print(test.get_circle_area(radius))

# import test_module

# print("# 메인의 __name__ 출력하기")
# print(__name__)
# print()

# def number_imput():
#     output= input("숫자 입력> ")
#     return float(output)

# def get_circumterence(radius):
#     return 2 * PI * radius

# def get_circumference(radius):
#     return PI * radius * radius

# if __name__=="__main":
#     print("get_circumference(10):", get_circumterence(10))
#     print("get_circle_area(10):", get_circle_area(10))

# variable_a = "a 모듈의 변수"
# variable_b = "b 모둘의 변수"

# import test)package.module_a as a
# import test_package.module_b as b

# print(a.variable_a)
# print(b.variable_b)

# __all__ = ["module_a", "module_b"]
# print("test_package를 읽어 들였습니다.")
# from test_package import *

# print(module_a.variable_a)
# print(module_b.variable_b)

# students = [
#     {"name":"윤인성", "korean":87, "math":98, "english":88, "science":95},
#     {"name":"연하진", "korean":92, "math":98, "english":96, "science":98},
#     {"name":"구지연", "korean":76, "math":96, "english":94, "science":90},
#     {"name":"나선주", "korean":98, "math":92, "english":96, "science":92},
#     {"name":"윤아린", "korean":95, "math":98, "english":98, "science":98},
#     {"name":"윤명월", "korean":64, "math":88, "english":92, "science":92}
# ]
# print("이름", "총점", "평균", sep="\t")
# for student in students:
#     score_sum = student["korean"] + student["math"]+\
#         student["english"]+student["science"]
#     score_average = score_sum/4
    
#     print(student["name"], score_sum, score_average, sep="\t")
    
# class Student:
#     def __init__(self, name, korean, math, enflish, science):
#         self.name = name
#         self.korean= korean
#         self.math= math
#         self.english= enflish
#         self.science=science
#     def get_sum(self):
#         return self.korean + self.math +\
#             self.english + self.science
#     def get_average(self):
#         return self.get_sum() /4
    
#     def to_string(self):
#         return "{}\t{}\t{}",format(\
#             self.name.\
#             self.get_sum(),\
#             self.get_average())

# Students = [
#     Student("윤이성", 87, 98, 88, 95),
#     Student("연하진",92, 98, 96, 98),
#     Student("구나진", 76, 96, 89, 90),
#     Student("나선주", 98, 92, 96, 92),
#     Student("윤아린", 95, 98, 98, 98),
#     Student("윤명월", 64, 88, 92, 92)
# ]
# print("이름", "총점", "평균", sep="\t")
# for student in Students:
#     print(student.to_string())

# for student in students:
#     print(student.name,student.korean,student.math,student.enflish,student.science)
# Student[0].name
# Student[0].korean
# Student[0].math
# Student[0].english
# Student[0].science

# class Student:
#     def __init__(self):
#         pass
# class Student(Human):
#     def __init__(self):
#         pass
    
# Student = Student()

# print("isinstance(student, Human):", (isinstance, Human))
# print("type(student( == Human:", type(student) == Human)

# class Student:
#     def study(self):
#         print("공부를 합니다.")
# class Teacher:
#     def teach(self):
#         print("학생을 가르칩니다.")
# classroom = [Student(), Student(), Teacher(), Student(), Student()]

# for person in classroom:
#     if isinstance(person, Student):
#         person.study()
#     elif isinstance(person, Teacher):
#         person.teach()


# class Student:
#     def __init__(self, name, korean, math, enflish, science):
#         self.name = name
#         self.korean= korean
#         self.math= math
#         self.english= enflish
#         self.science=science
        
#     def get_sum(self):
#         return self.korean + self.math +\
#             self.english + self.science
            
#     def get_average(self):
#         return self.get_sum()/4
    
#     def __str__(self):
#         return "{}\t{}\t{}".format(
#             self.name,
#             self.get_sum(),
#             self.get_average())
        
# Students = [
#     Student("윤이성", 87, 98, 88, 95),
#     Student("연하진", 92, 98, 96, 98),
#     Student("구나진", 76, 96, 89, 90),
#     Student("나선주", 98, 92, 96, 92),
#     Student("윤아린", 95, 98, 98, 98),
#     Student("윤명월", 64, 88, 92, 92)
# ]
# print("이름", "총점", "평균", sep="\t")
# for student in Students:
    # print(str(student))
    


# class Student:
#     def __init__(self, name, korean, math, enflish, science):
#         self.name = name
#         self.korean= korean
#         self.math= math
#         self.english= enflish
#         self.science=science
#     def get_sum(self):
#         return self.korean + self.math +\
#             self.english + self.science
#     def get_average(self):
#         return self.get_sum() /4
    
    # def to_string(self):
    #     return "{}\t{}\t{}",format(\
    #         self.name.\
    #         self.get_sum(),\
    #         self.get_average())
        
#     def __eq__(self, value):
#         return self.get_sum() == value.get_sum()
#     def __ne__(self, value):
#         return self.get_sum() != value.get_sum()
#     def __gt__(self, value):
#         return self.get_sum() > value.get_sum()
#     def __ge__(self, value):
#         return self.get_sum() >= value.get_sum()
#     def __lt__(self, value):
#         return self.get_sum() < value.get_sum()    
#     def __le__(self, value):
#         return self.get_sum() <= value.get_sum()
        
# Students = [
#     Student("윤이성", 87, 98, 88, 95),
#     Student("연하진", 92, 98, 96, 98),
#     Student("구나진", 76, 96, 89, 90),
#     Student("나선주", 98, 92, 96, 92),
#     Student("윤아린", 95, 98, 98, 98),
#     Student("윤명월", 64, 88, 92, 92)
# ]

# Student_a = Student("윤인성", 87, 98, 88, 95),
# Student_b = Student("연하진", 92, 98, 96, 98),

# print("student_a == student_b =", Student_a == Student_b)
# print("student_a != student_b =", Student_a != Student_b)
# print("student_a > student_b =", Student_a > Student_b)
# print("student_a >= student_b =", Student_a >= Student_b)
# print("student_a < student_b =", Student_a < Student_b)
# print("student_a <= student_b =", Student_a <= Student_b)


# class Student:
    
#     count = 0
#     Student= []
        
#     @classmethod
#     def print(cls):
#         print("------ 학생 목록 ------")
#         print("이름\t총점\t평균")
#         for student in cls.students:
#             print(str(student))
#         print("------- ------- -------")
#     def __init__(self, name, korean, math, enflish, science):
#         self.name = name
#         self.korean= korean
#         self.math= math
#         self.english= enflish
#         self.science=science
#         Student.count += 1
#         Student.students.append(self)      
#     def get_sum(self):
#         return self.korean+self.math+\
#             self.english+self.science
#     def get_average(self):
#         return self.get_sum()/4
#     def to_string(self):
#         return "{}\t{}\t{}",format(\
#             self.name,\
#             self.get_sum(),\
#             self.get_average())
# Student("윤인성", 87, 98, 88, 95)
# Student("연하진", 92, 98, 96, 98)
# Student("구나진", 76, 96, 89, 90)
# Student("나선주", 98, 92, 96, 92)
# Student("윤아린", 95, 98, 98, 98)
# Student("윤명월", 64, 88, 92, 92)

# Student.print()
# print()
# print("현재 생성된 총 학생 수는 {}명입니다.".format(Student.count))





# class Test:
#     def __init__(self, name):
#         self.name = name
#         print("{} - 생성되었습니다.">format(self.name))
#     def __del__(self):
#         print("{} - 파괴되었습니다.".format(self.name))
        
# a = Test("A")
# b = Test("B")
# c = Test("C")


# import math
# class Circle:
#     def __init__(self, radius):
#         self.__radius = radius
#     def get_curcumference(self):
#         return 2 ** math.pi * self.__radius
#     def get_area(self):
#         return math.pi * (self.__radius ** 2)
    
#     @property
#     def radius(self):
#         return self.__radius
#     @radius.setter
#     def radius(self, value):
#         self.__radius = value
#         if value <= 0:
#             raise TypeError("길이는 양의 숫자여야 합니다.")
#         self.__radius = value
        
    
# circle = Circle(10)
# print("원래 원의 반지름",circle.radius)
# circle.radius = 2
# print("변경된 원의 반지름", circle.radius)

# print("# 원의 둘레와 넓이를 구합니다.")
# print("원의 둘레: ",circle.get_curcumference())
# print("원의 넓이: ",circle.get_area())
# print(circle.get_radius())
# print(circle.set_radius(2))
# print("원의 둘레: ",circle.get_curcumference())
# print("원의 넓이: ",circle.get_area())


# class Parent:
#     def __init__(self):
#         self.value = "테스트"
#         print("Parent 클래스의 __init()__ 메소드가 호출되었습니다.")
#     def test(self):
#         print("Parent 클래스의 test() 메소드입니다.")
        
# class Child(Parent):
#     def __init__(self):
#         super().__init__()
#         print("Child 클래스의 __init__()메소드가 호출죄었습니다.")
            
#     child = Child()
#     child.test()
#     print(child.value)

# class CustomException(Exception):
#     def __init__(self):
#         super().__init__()
#         ㅇ
        
# raise CustomException

# result1 = 0
# result2 = 0

# def adder1(num):
#     global result1
#     result1 += num
#     return result1

# def adder2(num):
#     global result2
#     result2 += num
#     return result2

# adder1(1)
# print(result1)
# adder2(3)
# print(result2)
# adder1(5)
# print(result1)
# adder2(9)
# print(result2)

# class Calculator:
#     def __init__(self):
#         self.result = 0
        
#     def adder(self, num):
#         self.result += num
#         return self.result
    
# cai1 = Calculator()
# cai2 = Calculator()

# cai1.adder(3)
# cai2.adder(3)
# cai1.adder(5)
# cai2.adder(7)

# print(cai1.result)
# print(cai2.result)

# class Service:
#     def setname(self,name):
#         self.name = name
#     def sum(self,a,b):
#         result = a+b
#         print(f"{self.name}님 {a}+{b})는 {a+b}입니다.")
        
# pey = Service()
# pey.setname("홍길동")
# pey.sum(1,1)

# pal = Service()
# pal.setname("홍길균")
# pal.sum(3,5)


# class FourCal:
#     def setdata(self,first,second):
#         self.first = first
#         self.second = second
#     def sum(self):
#         result = self.first + self.second
#         return result
#     def mul(self):
#         result = self.first *self.second
#         return result

# a= FourCal()
# b = FourCal()
# a.setdata(4,2)
# b.setdata(3,7)
# print(a.sum())
# print(a.mul())
# print(a.sud())
# print(a.div())
# print(b.sum())
# print(b.mul())
# print(b.sud())
# print(b.div())
# print(type(a))
# a.setdata(4,2)

# print(a.first)
# print(a.second)

# b = FourCal()
# b.setdata(3,7)
# print(b.first)
# print(a.first)

# class HousePark:
#     lastname = "박"
#     def __init__(self,name):
#         self.fullname = self.lastname + name
#         def travel(self, where):
#             print(f"{self.fullname},{where}여행을 가다.")
#         def love(self, other):
#             print(f"{self.fullname},{other.fullname} 사랑에 빠졌네")
#         def fight(f"{self.fullname},{other.fullname} 싸우네")
#             print(f"{self.fullname},{other.fullname} 이혼했네")
#         def __add__(self, other):
#             print(f"{self.fullname},{other.fullname} 결혼했네")
            
# class Housekim(HousePark):
#     lastname = "김"
#     def travel(self, where, day):
#         print(f"{self.fullname},{where}여행 {day}일 가다")           
        
# pey = HousePark("응용")
# juliet = Housekim("줄리엣")
# pey.travel("부산")
# juliet
# pey.love(juliet)
# pey + juliet
# pey.travel("부산")
# pey = HousePark()

# print(pey.lastname)
# print(pes.lastname)

        
# juliet = Housekim("줄리엣")
# juliet.travel("독도",3)


# mylist = [lambda a,b:a+b]
 
# print(mylist[0](3,4))  

# def two_times(x):return x * 2

print(list(map(lambda a: a*2[1,2,3,4])))

def plus_one(x):return x +1
print(list(map(lambda a*2 )))

 