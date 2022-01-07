from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

def dcmotor(t, x):
    # Faulhaver사의 DC 모델 제원 (2232SR - 024SR)
    # 보고서의 3. 시뮬레이션에 해당하는 내용
    L = 710e-6      # [H], 인턱턴스
    J = 3.8e-7      # [kgm^2], 관성 모멘트
    R = 16.4        # [Ω], 모터 내부저항
    b = 0.0         # [Nm/rad/s], 마찰계수
    Kt = 3.21e-2    # [Nm/Wb/A], 토크 상수
    Ke = 3.2086e-2  # [Vs/rad/Wb], 역기전력
    v = 24          # [V], 전압


    # 상태 방정식에서 구한 행렬
    # 보고서(A2021014.docx)의 식 9
    A = (1/L*J) * np.array([[       0,             1     ],
                            [-(R*b + Kt*Ke), -(L*b + R*J)]])
    B = (1/L*J) * np.array([[0],
                            [Kt]])

    # 행렬곱을 위해 입력 x를 numpy array로 만들고,
    # (2, 1)의 shape을 가지도록 reshape
    X = np.array(x).reshape(-1, 1)

    # 마찬가지로 행렬곱을 위해 전압 v를 (2, 1)의
    # shape을 가지는 numpy array V로 생성
    V = np.array([[v],
                  [v]])

    # 보고서 식 10
    # 행렬 A: (2,2), 행렬 X: (2,1)의 shape이므로
    # 행렬곱 @ 연산하여 결과는 (2,1)
    # 행렬 B: (2,1), 행렬 V: (2,1)의 shape이므로
    # 스칼라곱 * 연산하여 결과는 (2,1)
    # 더하면 dx[0], dx[1]에 해당하는 dx 행렬을 얻음
    dx = A @ X + B * V

    # (2,)의 shape으로 return하기 위해 squeeze() 함수 적용
    return dx.squeeze()

if __name__ == '__main__':
    # 적용할 x1_init 생성 : (0, 10, 20, 30, 40)
    x1_range = np.arange(0, 50, 10)
    # 적용할 x2_init 생성 : (0, 10)
    x2_range = np.arange(0, 20, 10)
    # x1_range와 x2_range의 값을 조합하여 경우의 수 생성
    x_list = []
    for a in x1_range:
        for b in x2_range:
            x_list.append([a, b])

    # x_list에 있는 모든 x1_init, x2_init 조합에 대해 해석 진행
    for x1_init, x2_init in x_list:
        print(f'Now:: x1_init:{x1_init}, x2_init:{x2_init}')
        # 위에서 정의한 dcmotor 함수와, [시작 시간, 끝 시간], [x1 초기값, x2 초기값]을
        # scipy의 solve_ivp 함수에 전달
        sol = solve_ivp(dcmotor, [0, 2000], [x1_init, x2_init], max_step=0.1)
        # 시간과 그 시간에서의 해석값
        t, y = sol.t, sol.y
        # dx1, dx2
        dx1, dx2 = y[0], y[1]

        # 시간에 따른 dx1과 dx2를 plot
        plt.plot(t, dx1, t, dx2)
        plt.title('dx1, dx2', fontsize=20)
        plt.legend([f'dx1 (init:{x1_init})', f'dx2 (init:{x2_init})'],  fontsize=14)
        plt.xlabel('Time [0.1s]', fontsize=16)
        plt.ylabel('Value', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid()
        plt.show()

        # dx1과 dx2 비교 plot
        plt.plot(dx1, dx2)
        plt.title('dx1 - dx2 comparison', fontsize=20)
        plt.xlabel('dx1', fontsize=16)
        plt.ylabel('dx2', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid()
        plt.show()
