# Норми кодування для проєкту

## 1. Відступи та форматування
1.1 Відступи — 4 пробіли, без табуляцій.
1.2 Максимальна довжина рядка — 80 символів.
1.3 Порожній рядок перед визначенням нової функції або класу.

## 2. Іменування
2.1 Імена змінних — `snake_case`.
2.2 Імена функцій — `snake_case`.
2.3 Імена класів — `PascalCase`.
2.4 Імена констант — `UPPER_SNAKE_CASE`.

## 3. Структура файлів і модулів
3.1 Один модуль — одна відповідальність.
3.2 Максимум 300 рядків у файлі.

## 4. Коментарі та документація
4.1 Кожну функцію документувати у форматі docstring (reST або Google style).
4.2 Коментарі мають пояснювати «чому», а не «що».

## 5. Обробка помилок
5.1 Використовувати винятки (`raise`) для критичних ситуацій.
5.2 Не залишати `bare except` без вказівки типу винятку.

## 6. Тестування
6.1 Кожна функція має мінімум один unit-тест.
6.2 Використовувати pytest, з фікстурами для спільних налаштувань.

## 7. Інші правила
7.1 Не використовувати магічні числа — винести в константи.
7.2 Уникайте дублювання коду — виділяйте спільну логіку у функції.