using RedeNeural.exceptions;
using System;
using System.Collections.Generic;
using System.Text;

namespace RedeNeural
{
    public class Matriz
    {
        public int Linha { get; set; }
        public int Coluna { get; set; }
        public double[,] Valores { get; set; }

        public Matriz(int linha, int coluna)
        {
            Linha = linha;
            Coluna = coluna;
            Valores = new double[linha, coluna];
        }

        public void PopulaMatriz()
        {
            Random rd = new Random();
            
            for(int i = 0; i < Linha; i++)
            {
                for(int j = 0; j < Coluna; j++)
                {
                    var valor = rd.NextDouble();
                    Valores[i, j] = valor;
                }
            }
        }

        public static Matriz Multiplica(Matriz matrizA, Matriz matrizB)
        {
            if (matrizA.Coluna != matrizB.Linha)
            {
                throw new MatrizException("O número de colunas da primeira matriz deve ser igual número de linhas da segunda matriz.");
            }
            Matriz matrizAux = new Matriz(matrizA.Linha, matrizB.Coluna);

            var acumula = 0.0;
            //cada iteração representa uma linha da matriz A
            for (int linha = 0; linha < matrizA.Linha; linha++)
            {
                //em cada linha de A, itera nas colunas de B
                for (int coluna = 0; coluna < matrizB.Coluna; coluna++)
                {
                    //itera, ao mesmo tempo, entre os elementos da linha de A e da coluna de B
                    for (int i = 0; i < matrizA.Coluna; i++)
                    {
                        //acumula representa os valores que estávamos reservando
                        acumula = acumula + matrizA.Valores[linha, i] * matrizB.Valores[i, coluna];
                    }
                    //quando a execução está aqui, já se tem mais um elemento da matriz AB
                    matrizAux.Valores[linha, coluna] = acumula;

                    acumula = 0;
                }
            }

            return matrizAux;
        }

        public static Matriz Soma(Matriz matrizA, Matriz matrizB)
        {
            if (matrizA.Linha != matrizB.Linha || matrizA.Coluna != matrizB.Coluna)
            {
                throw new MatrizException("As matrizes precisam ter o mesmo tamanho, número de colunas e linhas iguais.");
            }
            Matriz matrizAux = new Matriz(matrizA.Linha, matrizA.Coluna);
            for (var i = 0; i < matrizA.Linha; i++)
            {
                for(var j = 0; j < matrizA.Coluna; j++)
                {
                    matrizAux.Valores[i, j] = matrizA.Valores[i, j] + matrizB.Valores[i, j];
                }
            }

            return matrizAux;
        }

        public static Matriz Subtrai(Matriz matrizA, Matriz matrizB)
        {
            if (matrizA.Linha != matrizB.Linha || matrizA.Coluna != matrizB.Coluna)
            {
                throw new MatrizException("As matrizes precisam ter o mesmo tamanho, número de colunas e linhas iguais.");
            }
            Matriz matrizAux = new Matriz(matrizA.Linha, matrizA.Coluna);
            for (var i = 0; i < matrizA.Linha; i++)
            {
                for (var j = 0; j < matrizA.Coluna; j++)
                {
                    matrizAux.Valores[i, j] = matrizA.Valores[i, j] - matrizB.Valores[i, j];
                }
            }

            return matrizAux;
        }

        public static Matriz Hadamard(Matriz matrizA, Matriz matrizB)
        {
            if (matrizA.Linha != matrizB.Linha || matrizA.Coluna != matrizB.Coluna)
            {
                throw new MatrizException("As matrizes precisam ter o mesmo tamanho, número de colunas e linhas iguais.");
            }
            Matriz matrizAux = new Matriz(matrizA.Linha, matrizA.Coluna);
            for (var i = 0; i < matrizA.Linha; i++)
            {
                for (var j = 0; j < matrizB.Coluna; j++)
                {
                    matrizAux.Valores[i, j] = matrizA.Valores[i, j] * matrizB.Valores[i, j];
                }
            }

            return matrizAux;
        }

        public static Matriz MultiplicaEscalar(Matriz matriz, double escalar)
        {
            Matriz matrizAux = new Matriz(matriz.Linha, matriz.Coluna);
            for (var i = 0; i < matriz.Linha; i++)
            {
                for (var j = 0; j < matriz.Coluna; j++)
                {
                    matrizAux.Valores[i, j] = matriz.Valores[i, j] * escalar;
                }
            }

            return matrizAux;
        }

        public static Matriz Transpoe(Matriz matriz)
        {
            Matriz matrizAux = new Matriz(matriz.Coluna, matriz.Linha);
            for (var j = 0; j < matriz.Coluna; j++)
            {
                for (var i = 0; i < matriz.Linha; i++)
                {
                    matrizAux.Valores[j, i] = matriz.Valores[i, j];
                }
            }

            return matrizAux;
        }

        /**
         * Esse método irá aplicar a função de ativação nos elementos de uma matriz 
         */
        public static Matriz Sigmoid(Matriz matriz)
        {
            Matriz matrizAux = new Matriz(matriz.Linha, matriz.Coluna);
            for (var i = 0; i < matriz.Linha; i++)
            {
                for (var j = 0; j < matriz.Coluna; j++)
                {
                    matrizAux.Valores[i, j] = 1 / (1 + Math.Exp(-matriz.Valores[i, j]));
                }
            }

            return matrizAux;
        }

        /**
         * Esse método irá aplicar a derivada da função de ativação nos elementos de uma matriz 
         */
        public static Matriz Dsigmoid(Matriz matriz)
        {
            for (var i = 0; i < matriz.Linha; i++)
            {
                for (var j = 0; j < matriz.Coluna; j++)
                {
                    matriz.Valores[i, j] = matriz.Valores[i, j] * (1 - matriz.Valores[i, j]);
                }
            }

            return matriz;
        }

        public static Matriz ArrayToMatriz(double[] array)
        {
            Matriz matrizAux = new Matriz(array.Length, 1);
            for(var i = 0; i < array.Length; i++)
            {
                matrizAux.Valores[i, 0] = array[i];
            }

            return matrizAux;
        }

        public static double[] MatrizToArray(Matriz matriz)
        {
            double[] array = new double[matriz.Linha];
            for(var i = 0; i < matriz.Linha; i++)
            {
                array[i] = matriz.Valores[i, 0];
            }

            return array;
        }
    }
}
