(() => {
  "use strict";

  const $ = (id) => /** @type {HTMLInputElement} */ (document.getElementById(id));

  const areaInput = $("area");
  const roomsInput = $("rooms");
  const ageInput = $("age");
  const btnCalc = /** @type {HTMLButtonElement} */ (document.getElementById("btnCalc"));

  const resultBox = document.getElementById("result");
  const priceText = document.getElementById("priceText");
  const detailText = document.getElementById("detailText");
  const tagsBox = document.getElementById("tags");

  /**
   * 一个简化版的房租估算模型：
   * - 基础单价：50 元/m²（可理解为普通城市的平均水平）
   * - 房间溢价：每多 1 个房间，单价 +5%
   * - 房龄折扣：前 5 年为“新”，单价 +8%；之后每 5 年 -6%，最多降到 0.6 倍
   * - 最终结果给出一个 ±10% 的合理区间
   */
  function estimateRent(area, rooms, age) {
    const baseP = 50; // 元/平

    // 面积处理
    const usableArea = Math.min(Math.max(area, 5), 500);

    // 房间溢价（2~4 是常见居住舒适区）
    const roomFactor = 1 + (rooms - 2) * 0.05;

    // 房龄折扣
    let ageFactor = 1.0;
    if (age <= 3) {
      ageFactor = 1.1;
    } else if (age <= 8) {
      ageFactor = 1.08;
    } else {
      // 每 5 年 -6%
      const blocks = Math.floor((age - 8) / 5);
      ageFactor = 1.08 * Math.pow(0.94, blocks);
      ageFactor = Math.max(ageFactor, 0.6);
    }

    const unitPrice = baseP * roomFactor * ageFactor;
    const center = unitPrice * usableArea;

    const low = Math.round(center * 0.9);
    const high = Math.round(center * 1.1);

    return { low, high, unitPrice: Math.round(unitPrice) };
  }

  function format(num) {
    return num.toLocaleString("zh-CN");
  }

  function describe(area, rooms, age) {
    const tags = [];
    if (area < 40) tags.push({ t: "面积偏小，适合单身或情侣", cls: "bad" });
    else if (area > 110) tags.push({ t: "面积较大，适合家庭使用", cls: "good" });

    if (rooms <= 2) tags.push({ t: "小户型", cls: rooms === 1 ? "bad" : "good" });
    else if (rooms >= 4) tags.push({ t: "多房间，卧室较多", cls: "good" });

    if (age <= 5) tags.push({ t: "新房 / 较新楼盘，溢价更高", cls: "good" });
    else if (age >= 20) tags.push({ t: "房龄偏大，价格一般会有折扣", cls: "bad" });

    return tags;
  }

  function calc() {
    const area = Number(areaInput.value);
    const rooms = Number(roomsInput.value);
    const age = Number(ageInput.value);

    if (!area || !rooms || age < 0 || Number.isNaN(age)) {
      alert("请完整填写：面积、房间数、房龄（年）。");
      return;
    }

    const { low, high, unitPrice } = estimateRent(area, rooms, age);
    resultBox.hidden = false;

    priceText.textContent = `${format(low)} ~ ${format(high)} 元/月`;
    detailText.textContent = `根据面积 ${area} m²、房间数 ${rooms}、房龄 ${age} 年，采用简化模型估算：参考单价约为 ${unitPrice} 元/m²·月，给出 ±10% 浮动的合理区间，仅作租金谈判参考。`;

    tagsBox.innerHTML = "";
    const tags = describe(area, rooms, age);
    for (const tag of tags) {
      const span = document.createElement("span");
      span.className = `tag ${tag.cls}`;
      span.textContent = tag.t;
      tagsBox.appendChild(span);
    }
  }

  btnCalc.addEventListener("click", calc);

  // 支持回车直接计算
  [areaInput, roomsInput, ageInput].forEach((el) => {
    el.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        calc();
      }
    });
  });
})();

