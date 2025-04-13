---
title: Java SPI 扩展
date: 2024-06-16 10:00:00 +0800
categories: [Java]
tags: [Java, Base]
---

## SPI 扩展

### 1. SPI 简介

SPI 全称 Service Provider Interface，是 Java 提供的一套用来被第三方实现或者扩展的 API。它可以用来启用框架扩展和替换组件。
SPI 的核心思想就是解耦，它可以用来解耦接口和实现，调用方对于接口的实现完全无感知，它只需要知道接口就足够了，具体的实现类可以由第三方来实现。

### 2. SPI 使用

#### 2.1 定义接口

```java
public interface Animal {
    void eat();
}
```

#### 2.2 实现接口

```java
public class Dog implements Animal {
    @Override
    public void eat() {
        System.out.println("dog eat bone");
    }
}

public class Cat implements Animal {
    @Override
    public void eat() {
        System.out.println("cat eat fish");
    }
}
```

#### 2.3 配置文件

在 `META-INF/services` 目录下创建一个以接口全限定名命名的文件，文件内容是实现类的全限定名。

```
com.example.spi.Animal
```

#### 2.4 加载实现类

```java
public class SpiTest {
    public static void main(String[] args) {
        ServiceLoader<Animal> animals = ServiceLoader.load(Animal.class);
        animals.forEach(Animal::eat);
    }   
}
```

### 3. SPI 扩展

#### 3.1 定义接口

```java
public interface Animal {
    void eat();
}
```

#### 3.2 实现接口

```java
public class Dog implements Animal {
    @Override
    public void eat() {
        System.out.println("dog eat bone");
    }
}

public class Cat implements Animal {
    @Override
    public void eat() {
        System.out.println("cat eat fish");
    }
}
```

#### 3.3 配置文件

在 `META-INF/services` 目录下创建一个以接口全限定名命名的文件，文件内容是实现类的全限定名。

```
com.example.spi.Animal
```

#### 3.4 加载实现类

```java
public class SpiTest {
    public static void main(String[] args) {
        ServiceLoader<Animal> animals = ServiceLoader.load(Animal.class);
        animals.forEach(Animal::eat);
    }
}

## 工作原理

1. ServiceLoader.load() 方法会加载指定接口的实现类，并按照配置文件中的顺序进行排序。
2. ServiceLoader 实现了 Iterable 接口，可以通过 foreach 循环遍历实现类。
3. ServiceLoader 会自动加载实现类，无需手动实例化。
4. ServiceLoader 会自动处理实现类的依赖关系，无需手动管理。

## 使用场景

### 在 Tomcat 中的使用

Tomcat 使用 SPI 机制加载各种 Connector，包括 HTTP、HTTPS、AJP 等。

### 在 Spring Boot 中的使用

Spring Boot 使用 SPI 机制加载各种自动配置类，包括数据源、缓存、消息队列等。

### 在 MyBatis 中的使用

MyBatis 使用 SPI 机制加载各种插件，包括分页插件、缓存插件等。

### 在 Dubbo 中的使用

Dubbo 使用 SPI 机制加载各种协议、序列化方式、负载均衡策略等。

## 总结

SPI 机制是一种非常灵活的插件化机制，可以方便地实现模块的扩展和替换。在使用 SPI 机制时，需要注意以下几点：

1. 接口和实现类必须放在不同的包中，否则无法加载。
2. 实现类必须实现接口，并使用 `@SPI` 注解标记。
3. 配置文件必须放在 `META-INF/services` 目录下，文件名必须为接口的全限定名。
4. 加载实现类时，可以使用 `ServiceLoader.load()` 方法，也可以使用 `ExtensionLoader.getExtensionLoader()` 方法。

通过 SPI 机制，我们可以轻松地实现模块的扩展和替换，提高代码的可维护性和可扩展性。