package com.example;

import com.example.dto.AmendOrderRequest;
import com.example.entity.ClientOrder;
import com.example.service.OrderService;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureWebMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.Optional;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@SpringBootTest
@AutoConfigureWebMvc
public class OrderControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private OrderService orderService;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    public void testAmendOrder_Success() throws Exception {
        // Given
        AmendOrderRequest request = new AmendOrderRequest();
        request.setOrderId(1L);
        request.setUserId("user123");
        request.setStatus("CONFIRMED");
        request.setAmount(new BigDecimal("1500.00"));

        ClientOrder mockOrder = new ClientOrder();
        mockOrder.setId(1L);
        mockOrder.setOrderNumber("ORD-001-2024");
        mockOrder.setClientId("CLIENT-001");
        mockOrder.setStatus("CONFIRMED");
        mockOrder.setAmount(new BigDecimal("1500.00"));
        mockOrder.setVersion(2);

        when(orderService.amendOrder(any(AmendOrderRequest.class)))
                .thenReturn(com.example.dto.AmendOrderResponse.success(
                        "Order amended successfully",
                        "test-lock-token",
                        LocalDateTime.now().plusMinutes(5),
                        mockOrder
                ));

        // When & Then
        mockMvc.perform(post("/api/orders/amend")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.success").value(true))
                .andExpect(jsonPath("$.message").value("Order amended successfully"))
                .andExpect(jsonPath("$.lockToken").value("test-lock-token"))
                .andExpect(jsonPath("$.order.id").value(1))
                .andExpect(jsonPath("$.order.status").value("CONFIRMED"));
    }

    @Test
    public void testAmendOrder_OrderAlreadyLocked() throws Exception {
        // Given
        AmendOrderRequest request = new AmendOrderRequest();
        request.setOrderId(1L);
        request.setUserId("user123");

        when(orderService.amendOrder(any(AmendOrderRequest.class)))
                .thenReturn(com.example.dto.AmendOrderResponse.error(
                        "Order is currently locked by user user456 until 2024-01-15T10:25:00",
                        "ORDER_ALREADY_LOCKED"
                ));

        // When & Then
        mockMvc.perform(post("/api/orders/amend")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isConflict())
                .andExpect(jsonPath("$.success").value(false))
                .andExpect(jsonPath("$.errorCode").value("ORDER_ALREADY_LOCKED"));
    }

    @Test
    public void testGetOrderById_Success() throws Exception {
        // Given
        ClientOrder mockOrder = new ClientOrder();
        mockOrder.setId(1L);
        mockOrder.setOrderNumber("ORD-001-2024");
        mockOrder.setClientId("CLIENT-001");
        mockOrder.setStatus("PENDING");
        mockOrder.setAmount(new BigDecimal("1000.00"));

        when(orderService.getOrderById(1L))
                .thenReturn(Optional.of(mockOrder));

        // When & Then
        mockMvc.perform(get("/api/orders/1"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id").value(1))
                .andExpect(jsonPath("$.orderNumber").value("ORD-001-2024"))
                .andExpect(jsonPath("$.clientId").value("CLIENT-001"));
    }

    @Test
    public void testGetOrderById_NotFound() throws Exception {
        // Given
        when(orderService.getOrderById(999L))
                .thenReturn(Optional.empty());

        // When & Then
        mockMvc.perform(get("/api/orders/999"))
                .andExpect(status().isNotFound());
    }

    @Test
    public void testReleaseLock_Success() throws Exception {
        // Given
        when(orderService.canAmendOrder(1L, "user123"))
                .thenReturn(true);

        // When & Then
        mockMvc.perform(delete("/api/orders/locks/test-token")
                        .param("userId", "user123"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.success").value(true));
    }

    @Test
    public void testCanAmendOrder_Success() throws Exception {
        // Given
        when(orderService.canAmendOrder(1L, "user123"))
                .thenReturn(true);

        // When & Then
        mockMvc.perform(get("/api/orders/1/can-amend")
                        .param("userId", "user123"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.success").value(true));
    }

    @Test
    public void testCanAmendOrder_Conflict() throws Exception {
        // Given
        when(orderService.canAmendOrder(1L, "user123"))
                .thenReturn(false);

        // When & Then
        mockMvc.perform(get("/api/orders/1/can-amend")
                        .param("userId", "user123"))
                .andExpect(status().isConflict())
                .andExpect(jsonPath("$.success").value(false))
                .andExpect(jsonPath("$.errorCode").value("ORDER_CANNOT_BE_AMENDED"));
    }
} 