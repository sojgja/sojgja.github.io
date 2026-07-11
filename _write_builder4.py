import os

path = r'F:\git\sojgja.github.io\docs\series\builder.md'

content = '''
## Uu va nhuoc diem

| Uu diem | Nhuoc diem |
|---------|-----------|
| **Kiem soat tung buoc**: Xay dung object phuc tap theo tung buoc ro rang | **Code quantity**: Can nhieu class (Builder interface + nhieu ConcreteBuilders) |
| **Tai su dung**: Cung process xay dung cho nhieu representation | **Complexity**: Pattern co the qua phuc tap cho object don gian |
| **Single Responsibility**: Tach construction khoi business logic | **Mutable builder**: Builder thuong co state — can reset giua cac lan build |
| **Fluent interface**: Code goi ro rang, de doc, de maintain | **Director rigid**: Director co the qua cung nhac neu process thay doi |
| **Validation**: Co the validate truoc khi tra ve product | **Thread-safety**: Builder khong an toan cho da luong |
| **Immutability**: De dang tao immutable product | **Memory overhead**: Builder luu intermediate state |
| **Open/Closed**: Them representation khong sua code cu | **Learning curve**: Developer moi can hieu Builder + Director relationship |

## Ket luan

Builder la pattern ly tuong khi ban can xay dung object phuc tap voi nhieu buoc va nhieu bieu dien. **Golden rule**: Neu constructor cua ban co 5+ tham so (dac biet la optional parameters), hoac neu object co the duoc tao theo nhieu cach khac nhau — do la dau hieu ban can Builder.

Hay nho su khac biet quan trong:
- **Factory Method / Abstract Factory**: Tao object *ngay lap tuc* (goi factory, nhan product).
- **Builder**: Tao object *qua nhieu buoc* (goi method A, method B, build).

Builder dac biet manh khi ket hop voi **Fluent Interface** (method chaining) va **Director** (dong goi quy trinh xay dung). Trong thuc te, ban thuong thay Builder trong:
- **Query builders**: SQL, Elasticsearch, Django ORM.
- **Document builders**: PDF, HTML, Excel, Word.
- **Configuration builders**: Dockerfile, Kubernetes manifests, CI/CD pipelines.

Khong phai luc nao cung can Director. Neu client code co the tu dieu khien cac buoc (va muon su linh hoat do), chi can Builder voi fluent interface la du. Director chi thuc su can khi ban muon chuan hoa quy trinh xay dung.
'''

with open(path, 'a', encoding='utf-8') as f:
    f.write(content)

print(f'Part 4 appended. Final size: {os.path.getsize(path)} bytes')
