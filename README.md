Проект для парсинга изображений

Для запуска сервера необходимо запустить файл main.py
Перед локальным запуском необходимо установить пакеты poppler и PyMuPDF

У сервера доступен WEB интерфейст по адресу http://127.0.0.1:3000

В корневой директории находится папка files в которой располагаются примеры файлов для парсинга.

Для тестирования эндпоинтов есть файл client.py
В файле тестов можно выбрать файл для отправки, результаты будут сохраняться в директорию csvs

Что умеет данный сервис:
 1. Преобразовывать файл pdf в набор jpeg файлов (качество преобразования зафиксировано в коде)
 2. Выполнять ручной поворот изображения
 3. Выполнять автоповорот изображения (по умолчанию считается, что таблица размещена в альбомном формате)
 4. Размечать на изображении линии таблицы (горизонтально - вертикальная сетка)
