# task_manager.py 

 

class Task: 

    def __init__(self, task_id, description): 

        self.task_id = task_id 

        self.description = description 

        self.is_completed = False 

 

    def complete(self): 

        self.is_completed = True 

 

    def __repr__(self): 

        status = "Done" if self.is_completed else "Pending" 

        return f"Task({self.task_id}, {self.description}, {status})" 

 

 

class TaskManager: 

    def __init__(self): 

        self.tasks = {}   # Dictionary to store tasks with task_id as key. 

        self.next_id = 1 

 

    def add_task(self, description): 

        task = Task(self.next_id, description) 

        self.tasks[self.next_id] = task 

        self.next_id += 1 

        return task 

 

    def remove_task(self, task_id): 

        if task_id in self.tasks: 

            del self.tasks[task_id] 

        else: 

            raise ValueError("Task not found") 

 

    def complete_task(self, task_id): 

        if task_id in self.tasks: 

            self.tasks[task_id].complete() 

        else: 

            raise ValueError("Task not found") 

 

    def get_all_tasks(self): 

        return list(self.tasks.values()) 