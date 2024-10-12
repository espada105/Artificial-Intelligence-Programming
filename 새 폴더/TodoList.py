class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def addList(self, task):
        new_node = Node(task)
        if self.head is None:
            self.head = new_node
        else:
            current = self.head
            while current.next is not None:
                current = current.next
            current.next = new_node

    def showList(self):
        if self.head is None:
            print("Todo List 목록이 없습니다.")
        else:
            current = self.head
            while current is not None:
                print("- " + current.data)
                current = current.next

    def removeList(self, task):
        current = self.head
        prev = None

        while current is not None:
            if current.data == task:
                if prev is None:
                    self.head = current.next
                else:
                    prev.next = current.next
                print(f"'{task}'를 제거했습니다.")
                return
            prev = current
            current = current.next

        print(f"'{task}' Todo List 목록을 찾을 수 없습니다. 정확하게 입력해주세요.")

todo_list = LinkedList()

while True:
    print("\nTodo List: 번호를 입력하면 해당 메뉴를 실행할 수 있습니다.")
    print("1. Todo List 추가")
    print("2. Todo List 제거")
    print("3. Todo List 목록 보기")
    print("4. Todo List 종료")
    choice = input("선택한 메뉴: ")

    if choice == '1':
        task = input("추가할 Todo List: ")
        todo_list.addList(task)
    elif choice == '2':
        task = input("제거할 Todo List: ")
        todo_list.removeList(task)
    elif choice == '3':
        print("\nTodo List 목록:")
        todo_list.showList()
    elif choice == '4':
        print("TodoList를 종료합니다.")
        break
    else:
        print("잘못된 번호를 입력했습니다. 정확하게 입력해주세요.")