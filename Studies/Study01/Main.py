# http://web.archive.org/web/20170221220235/http://users.rcn.com/python/download/Descriptor.htm
class Main(object):
    """description of class"""


    def __init__(self, initval=None, name='var'):
            self.val = initval
            self.name = name

    def __get__(self, obj, objtype):
        print ('Retrieving', self.name)
        return self.val

    def __set__(self, obj, val):
        print ('Updating' , self.name)
        self.val = val

    def getx(self):
        return self.__x
    def setx(self, value):
        self.__x = value
    def delx(self):
        del self.__x
    x = property(getx, setx, delx, "I'm the 'x' property.")

m = Main()

print(m.name)
m.x = 100

print(m.x)
