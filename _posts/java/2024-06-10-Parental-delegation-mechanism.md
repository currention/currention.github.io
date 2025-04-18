---
title: Java 双亲委派机制
date: 2024-06-10 15:00:00 +0800
categories: [Java]
tags: [Java, Java Basic]
---

双亲委派机制是Java类加载器的一种机制，用于确保Java类加载器在加载类时遵循一定的顺序，避免重复加载和加载错误。本文将介绍双亲委派机制的基本原理、实现方式以及其优缺点。

## 基本原理

双亲委派机制的基本原理是，当一个类加载器加载一个类时，它会首先将加载任务委派给其父类加载器去完成。如果父类加载器能够成功加载该类，则直接返回；否则，再由当前类加载器去加载该类。这种机制确保了Java类加载器在加载类时遵循一定的顺序，避免了重复加载和加载错误。

## 类加载器分类

在Java中，类加载器分为以下几种：
- Bootstrap ClassLoader：启动类加载器，负责加载Java核心类库，如rt.jar等。
- Extension ClassLoader：扩展类加载器，负责加载Java扩展类库，如javax等。
- Application ClassLoader：应用程序类加载器，负责加载用户自定义的类。

## 优缺点

双亲委派机制的优点是：
- 避免了重复加载和加载错误。
- 确保了Java类加载器在加载类时遵循一定的顺序，提高了安全性。

双亲委派机制的缺点是：
- 加载速度较慢，因为需要先委派给父类加载器去加载。
- 不适合动态加载类的情况，因为动态加载类需要绕过双亲委派机制。

# 工作原理
双亲委派机制的工作原理如下：
1. 当一个类加载器加载一个类时，它会首先将加载任务委派给其父类加载器去完成。
2. 如果父类加载器能够成功加载该类，则直接返回；否则，再由当前类加载器去加载该类。
3. 如果当前类加载器也无法加载该类，则会抛出ClassNotFoundException异常。

# 如何打破双亲委派机制
双亲委派机制可以被打破，通过自定义类加载器来实现。自定义类加载器可以通过重写ClassLoader的loadClass方法来打破双亲委派机制。