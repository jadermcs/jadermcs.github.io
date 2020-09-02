---
title: The Problem of Induction
date: 2020-09-03 12:32:00
tags: [induction, epistemology]
author: Jader Martins
---
The cause and effect framework is a natural way of human thinking

# Epistemology
There's a branch in philosophy concerning the validity, nature, and other issues in
the scientific reasoning, this branch is known as Epistemology, from Greek "study of the
knowledge",

In this post we will reason about the inferencial aspect of reasoning, it is how
we know that a phenomena A causes a phenomena B, in formal syntax ($A \rightarrow B$)?
https://en.wikipedia.org/wiki/Epistemology
https://plato.stanford.edu/entries/epistemology/

## Deduction
In fields like Philosophy, Mathematics, and some times Physics, the corpus of knowledge
resides in axioms, theorems and pure forms. So every conclusion could be obtained by a
natural deduction over the initial truths.

https://en.wikipedia.org/wiki/Theory_of_forms

      1. forAll(x, Gx -> Hx)      P            // premise
      2. forAll(x, Hx -> ~Ix)     P            // premise
      3. Ga -> Ha                  1 E.forAll  // eliminate forAll in 1
      4. Ha -> ~Ia                 2 E.forAll  // eliminate forAll in 2
      5. Ga -> ~Ia                 3,4 SH      // hypothetic silogism in 3 and 4
      6. forAll(x, Gx -> ~Ix)      5 I.forAll  // introduces forAll

For example, Einstein with a pure mathematical reasoning deduced the "$e = mc^2$" formula.

## Indutivo
For other bodies of knowledge, like Psychology, Economy, Epidemiology, is not possible
(at least for know) to obtain perfect information about the studied object. There's
instrumental erros, randomized outputs given human dependecy, etc.

Observação de corpos celestes: F = G*M*m/r^2

    Ex: x conduz eletricidade
    Mx: x é um metal.

    1. Ef ^ Mf // iron conducts eletricity and is a metal
    2. Ec ^ Mc // copper conducts eletricity and is a metal
    3. Eg ^ Mg // gold conducts eletricity and is a metal
    4. Es ^ Ms // silver conducts eletricity and is a metal
    ...

So, every metal conducts eletricity...

        forAll(x, Mx -> Ex)

Or, every conductor of electricity is a metal??

        forAll(x, Ex -> Mx)

Water conducts eletricity, but isn't a metal.

Metals conducts eletricity, which is truth for this case, but can be coincidence for other
correlations.


If A and B are correlated, how we know if A causes B ($A \rightarrow B$), or B causes A
($B \rightarrow A$), or they are just mutually causes by a third part, Z ($Z \rightarrow A
\land Z \rightarrow B$)??

https://plato.stanford.edu/entries/counterfactuals

## The "Problem of Induction"

It is a famous [epistemological problem](https://plato.stanford.edu/entries/induction-problem/)
the begins with the "Enquiry on Human Understading" written by David Hume in XXX
where he YYYYY
How we know if a hypothesis is general enough?

, later in XXX, Immanuel Kant in his "Critique of Pure Reason" argues that
YYYY.

Firstly we need  to formulate a hypothesis
    - Impossível ter certeza no raciocínio indutivo.

    - Busca pela conclusão mais provável e por contra exemplos.

    - Infinitas funções podem explicar um par (observações,propriedades)
## Probability at Rescue
Hume horns

## A Perfect Induction System
Given that we have the data that really describes and observable phenomena, this framework
garantees to find the best (most probable) hypothesis.






https://www.gwern.net/Correlation
https://www.gwern.net/Causality

*** O "Problema da Indução"

*** Navalha de Ockham e a Indução Universal de Solomonoff

    - Se o mundo pode ser descrito por programas de computador, programas menores são
      mais prováveis de serem tal descrição. (Teoria da Probabilidade Algorítmica)

    - Indução Universal não é computável (Problema da Parada).

    - AIXI -> AGI

    - Certas questões do aprendizado de máquina só podem ser explicadas pela teoria
       do aprendizado computacional, não pelo aprendizado estatístico.

    - Computação tem importância tanto prática (engenharia) quanto teórica para o
       aprendizado de **máquina** (limites da máquina).

    - Recursos são finitos, tempo, espaço, custos, Nubank por exemplo implementa suas
       máquinas de inferência em Clojure/Spark, R não é viável.

    - SVMs tem uma teoria estatística brilhante (Dimensão VC, kernels R^infinito) mas
       entraram em desuso pelo alto custo computacional e resultados práticos limitados.

    - Teoria do "Aprendível"/Provavelmente Aproximadamente Correto, podemos aprender
       um função de criptografia se tivéssemos os pares (rawtext, encryptedtext)?

    - Podemos resolver quais classes de problemas  com aprendizado de máquina? P? NP? BQP?

    - Computadores tem precisão finita, não operamos por Reais mas sim por Reais
       Computáveis (conjunto contável e na prática finito), o que gera implicações e
       limitações teóricas.

* Aprendizado de Máquina
  No Aprendizado de máquina acreditamos que as características descritivas dos dados
  podem ser usadas para inferir alguma propriedade a respeito deles.

  "Tamanhos da pétala de uma Iris podem determinar de qual especie ela."

  Tamanho e largura da pétala, tamanho e largura da sépala.

	       X                          y
      |7.3  2.9  6.3  1.8|               |2|
      |4.8  3.   1.4  0.3|               |0|
      |4.6  3.1  1.5  0.2|   -> f(X) ->  |0|
      |6.1  2.8  4.   1.3|               |1|
      |5.6  2.8  4.9  2. |               |2|

  Como descobrir a f (f é uma função computável)??

  #+BEGIN_SRC python
    def algoritmo(X, y):
        f = None
        minimo_erro = MAX_FLOAT
        for h in [lambda x: x**2+5,
                  lambda x:sin(x)/log(x),
                  lambda x: sqrt(x)^pi,
                  # ...
                  lambda x: 42]:
            erro = computa_erro(h(X), y)
            if erro < minimo_erro:
                minerro = erro
                f = h
        return f
  #+END_SRC

  Arvores de decisão: Dividir e conquistar ou greedy search, {arvores que biparticionam os
	 dados pelos atributos}.

  SVMs: Programação quadrática, {hiperplanos da forma W*x.T+w_0 = 0}

  Redes Neurais: Otimização por gradiente descendente, {TODAS as funções computáveis, aproximador
		universal}.

  Algoritmos F tem diversos f_n's em que um deles é escolhido de forma a melhor explicar
  o problema. Porém, não há garantias que você não tenha dado "sorte".

  Tirar 5 vezes cara na moeda significa que a moeda só dá cara? Amostra != População.

  #+BEGIN_SRC python
    def f(X):
        if X == [7.3, 2.9, 6.3, 1.8]:
            return [2]
        if X == [4.8, 3., 1.4, 0.3]:
            return [0]
        # ...
        else:
            return "???"
  #+END_SRC

  É uma boa função??
      Se dentro dos dados que conheço, sim, caso contrário, é horrível.

  Há garantias quanto a generalização usando os algoritmos citados acima se respeitado o método e
  o fenomeno gerador dos dados for independente e identicamente distribuído.  (Vapnik-Chervonenkis)

* Meta-Aprendizado
insight: Usar conhecimento "meta" para resolver problemas a nível meta.

** Problema de Recomendação
Jader (algoritmo meta), observa as especialidades dos orientados do Prof.Luis,
(algoritmos base).

- Fulano: "Estou com um problema de desenvolvimento web, pode me ajudar?"
- Jader:  "Eu não posso, mas o Thales pode!"

irisdata    -> mfe(XX) -> [4.8  3.   1.4  0.3]  |  melhor classificador -> random forest=F_2
bostondata  -> mfe(XX) -> [5.6  2.8  4.9  2. ]  |  melhor classificador -> svm=F_1
  .                                 .           |                           .
  .                                 .           |                           .
  .                                 .           |                           .
  .                                 .           |                           .

         X                           y
|4.8  3.   1.4  0.3|               |F_2|
|6.1  2.8  4.   1.3|               |F_1|
|4.6  3.1  1.5  0.2|   -> f(X) ->  |F_2|
|7.3  2.9  6.3  1.8|               |F_1|
|5.6  2.8  4.9  2. |               |F_2|

- Como construir o MFE(), quais atributos me geram informação suficiente para aprender a tarefa??
- Como posso aprender f?? Em quais casos é possível aprender f?? (artigo mtlbias)
- Quais problemas de recomendação de algoritmo posso modelar para serem aprendidos??
- ML na 4dimensão (meta), campo relativamente novo e com infinitas possibilidades.
        (foto do astronauta do interestelar no tesseract)

* Extra
  - Teorias do aprendizado computacional se estendem a problemas muito mais complexos que f:X->y
  - Identifição de Linguagem no Limite.

