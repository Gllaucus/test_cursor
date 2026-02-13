(() => {
  "use strict";

  // TODO: 如果你的 Python 服务不是部署在同一个域名/端口，请改成完整 URL，
  // 比如 "http://127.0.0.1:8000/api/rent"
  const API_URL = "/api/rent";

  const $ = (id) => /** @type {HTMLInputElement} */ (document.getElementById(id));

  const areaInput = $("area");
  const roomsInput = $("rooms");
  const ageInput = $("age");
  const btnCalc = /** @type {HTMLButtonElement} */ (document.getElementById("btnCalc"));

  const resultBox = document.getElementById("result");
  const priceText = document.getElementById("priceText");
  const detailText = document.getElementById("detailText");
  const errorText = document.getElementById("errorText");

  function showError(msg) {
    resultBox.hidden = false;
    errorText.hidden = false;
    errorText.textContent = msg;
  }

  async function calc() {
    const area = Number(areaInput.value);
    const rooms = Number(roomsInput.value);
    const age = Number(ageInput.value);

    if (!area || !rooms || age < 0 || Number.isNaN(age)) {
      alert("请完整填写：面积、房间数、房龄（年）。");
      return;
    }

    btnCalc.disabled = true;
    btnCalc.textContent = "正在请求后端...";
    errorText.hidden = true;
    errorText.textContent = "";

    try {
      const resp = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ area, rooms, age }),
      });

      if (!resp.ok) {
        const text = await resp.text();
        showError(`后端返回错误状态 ${resp.status}：${text.slice(0, 120)}`);
        return;
      }

      /** @type {{rent?: number} & Record<string, any>} */
      const data = await resp.json();
      if (typeof data.rent !== "number" || Number.isNaN(data.rent)) {
        showError("后端返回的数据里没有有效的 rent 字段，请检查接口返回格式。");
        return;
      }

      resultBox.hidden = false;
      priceText.textContent = `${data.rent.toLocaleString("zh-CN")} 元/月`;
      detailText.textContent = `由后端 Python 函数根据面积 ${area} m²、房间数 ${rooms}、房龄 ${age} 年直接计算得到的估算月租金。`;
    } catch (err) {
      showError(`请求后端失败：${err}`);
    } finally {
      btnCalc.disabled = false;
      btnCalc.textContent = "调用后端估算";
    }
  }

  btnCalc.addEventListener("click", () => {
    calc();
  });

  [areaInput, roomsInput, ageInput].forEach((el) => {
    el.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        calc();
      }
    });
  });
})();

